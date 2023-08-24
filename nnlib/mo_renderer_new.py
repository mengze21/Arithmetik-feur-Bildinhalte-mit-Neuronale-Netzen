__version__ = "mrn_11"

import math
import six
import warnings
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.optimize as opti
from abc import ABCMeta, abstractmethod

from .support_lib import MOD_SETTINGS, NNLibUsageError, Breakout
from .support_lib import ang_index
from .render_tooling import RenderInterface, register_render_if


def _general_transition(old):
    """Calculates the transitions between sides inside a pixel."""
    if -0.5 < old[1] < 0.5:
        newx = old[0]
    else:
        newx = -old[1]
    if -0.5 < old[0] < 0.5:
        newy = old[1]
    else:
        newy = old[0]
    return (newx, newy)


def integrate_border(rlist, pixel):
    """This function gets called to deliver the area covered for
    every pixel on the border of the object.
    """

    def ashift(to_shift):
        return (to_shift[0] + pixel[0], to_shift[1] + pixel[1])

    area = 0.0
    up = False
    if MOD_SETTINGS['DEBUG']:
        cycle_counter = 0
    while rlist:
        current = rlist[0][2]
        while True:
            trans = _general_transition(current)
            c_distance = abs((trans[0] - current[0]) + (trans[1] - current[1]))
            distance = 0.0
            next = None
            if MOD_SETTINGS['DEBUG']:
                cycle_counter += 1
                if cycle_counter > 50:
                    raise NNLibUsageError("Maximum cycles on pixel border "
                                          "reached. Something seems to be "
                                          "wrong with the border processing. "
                                          "Error arose on pixel (%i,%i)."
                                          % (pixel[0], pixel[1]))
            for i in range(len(rlist)):
                if rlist[i][1] == trans:
                    curr_dist = abs((trans[0] - rlist[i][0][0])
                                    + (trans[1] - rlist[i][0][1]))
                    if c_distance >= curr_dist > distance:
                        distance = curr_dist
                        next = i
            if next is not None:
                area += (_integrate_straight_line(ashift(current),
                                                  ashift(rlist[next][0]))
                         + rlist[next][3])
                current = rlist[next][2]
                rlist.pop(next)
                if next == 0:
                    break
            else:
                area += _integrate_straight_line(ashift(current),
                                                 ashift(trans))
                up = up or (current[0] - trans[0] == 1.0)
                current = trans
    return area, up


def make_relative_list(alist, pixel):
    """Creates points list relative to the pixel center from points
    list in absolute coordinates, as well as some additional info for
    easy comprehension.  Additionally detects and passes back relevant
    information about contributions that are completely internal to a
    given pixel, and excludes its entries from the list it returns.
    """
    rlist = []
    internal_area = 0.0
    for item in alist:
        if item[0]:
            rel_start = (item[0][0] - pixel[0], item[0][1] - pixel[1])
            mod_start = _general_transition(rel_start)
            rel_end = (item[1][0] - pixel[0], item[1][1] - pixel[1])
            rlist.append((rel_start, mod_start, rel_end, item[2]))
        else:
            internal_area += item[2]
    return rlist, internal_area


def _straight_cross(sp1, sp2, dir1, dir2):
    """Calculates the cross points of two straight lines.  sp is the
    starting point, dir is the direction. The return values contain the
    factors on the directions where the lines intersect.  This function
    does not perform any error handling.  Specifically, singular
    situations (parallel/identical lines) will cause a
    numpy.linalg.LinAlgError that needs to be handled by the caller.
    """
    a = np.array(((dir1[0], -dir2[0]),
                  (dir1[1], -dir2[1])))
    b = np.array((sp2[0] - sp1[0], sp2[1] - sp1[1]))
    return np.linalg.solve(a, b)


def _integrate_straight_line(begin, end):
    """Delivers the integral for a straight line between the two points
    provided.
    """
    return begin[0] * end[1] - begin[1] * end[0]


def _scalar_product(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]


class Node:
    """In the context of the renderer, objects are defined as sequences
    of nodes and edges.  This class represents the nodes, i.e., points
    where edges intersect.  The class provides the tooling to perform
    border tracing when merging objects and rasterizing.
    """

    def __init__(self, coords):
        self.coords = coords
        self.ins = [None, None]
        self.outs = [None, None]
        self.in_use = [False, False]

    def check(self, index):
        return self.in_use[index]

    def register_edges(self, line_in, line_out):
        if self.ins[0] is None:
            self.ins[0] = line_in
            self.outs[0] = line_out
        elif self.ins[1] is None:
            self.ins[1] = line_in
            self.outs[1] = line_out
        else:
            raise NNLibUsageError("Cannot register three crossing lines in "
                                  "one node.")

    def replace_edge(self, old, new):
        try:
            for c_list in (self.outs, self.ins):
                for edge_id in range(2):
                    if c_list[edge_id] == old:
                        c_list[edge_id] = new
                        raise Breakout()
            else:
                raise ValueError("The edge that should have been replaced "
                                 "was not found. This strongly indicates "
                                 "misuse of some description.")
        except Breakout:
            pass

    def get_edge(self, edge_in):
        if edge_in in {0, 1}:
            self.in_use[edge_in] = True
            out = int(not edge_in)
            return self.outs[out]
        elif self.outs[1] is None:
            return self.outs[0]
        elif self.ins[0] == edge_in:
            self.in_use[0] = True
            return self.outs[1]
        elif self.ins[1] == edge_in:
            self.in_use[1] = True
            return self.outs[0]
        else:
            raise NNLibUsageError("Node.trace must be called giving either "
                                  "an edge that is registered as an incoming "
                                  "edge or an integer indicating one of the "
                                  "outputs (0 or 1).")

    def get_distance(self, point, edge_in):
        """This function serves to enable the VectorLines'
        'check_inside' method.  It returns the euclidean distance of a
        given point from this node as well as a boolean value that is
        True if the point is to the left of the node (meaning 'on the
        inside' for additive objects) and False if it is to the right.
        Nodes inherit their sense of direction from their associated
        edges, requiring them to give appropriately oriented gradients.
        """
        try:
            edge_out = self.get_edge(edge_in)
        except NNLibUsageError:
            raise NNLibUsageError("Node.get_distance requires a registered "
                                  "incoming edge to be given so it knows "
                                  "which gradients to consider.")
        pointer = (point[0] - self.coords[0], point[1] - self.coords[1])
        dist = math.sqrt((self.coords[0] - point[0]) ** 2
                         + (self.coords[1] - point[1]) ** 2)
        grad_1 = edge_in.get_gradient(self.coords)
        grad_2 = edge_out.get_gradient(self.coords)
        a = np.array(((grad_1[0], grad_2[0]),
                      (grad_1[1], grad_2[1])))
        b = np.array(pointer)
        try:
            params = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            # We do not have to deal with singular cases because they
            # will be caught in the adjacent edge's version of this
            # function.  Therefore, we set 'params' to dummy values so
            # the function can go on as normal, but will yield 'None'
            # in the end.
            params = (-1.0, 1.0)
        if (params[0] < 0.0) == (params[1] < 0.0):
            if params[0] < 0.0:
                dist = -dist
        else:
            # The distance is set to inf if the base point for the
            # distance is not within the bounds of the edge.
            dist = float('inf')
        return dist


@six.add_metaclass(ABCMeta)
class Edge:
    """Base class for edges (straights and arcs).  Provides the tooling
    to perform outline tracing on them and defines some method stubs to
    be implemented by children of this class.
    """

    def __init__(self, source, startpoint, endpoint):
        # Register the source shape
        self.source = source
        self.startpoint = startpoint
        self.endpoint = endpoint
        # The list of splits to perform on this edge.  Gets filled when
        # joining objects.
        self.splits = []
        self.first_to_draw = False
        self.firstpoint = None

    def trace(self, edge_list, initial_edge):
        """Traces the shape of a newly defined VectorLine object."""
        # Get the next edge from the edge's own endpoint
        next = self.endpoint.get_edge(self)
        x_diff = self.endpoint.coords[0] - self.startpoint.coords[0]
        y_diff = self.endpoint.coords[1] - self.startpoint.coords[1]
        # CHANGED: If, while tracing, it is detected that a line is of
        # length (close to) zero (as defined by the following 'if'
        # condition), it will not append itself to the edge list and
        # will additionally replace references to itself with ones to
        # the next edge in line, thereby effectively deleting itself.
        if abs(x_diff) + abs(y_diff) < 1e-8:
            self.startpoint.replace_edge(self, next)
            next.startpoint = self.startpoint
            # In the special case that the first edge gets eliminated,
            # the initial edge needs to be set to 'next' for subsequent
            # traces as well so the breakout criterion continues to
            # work.  Since this conflicts with the normal procedure of
            # this method, we need to return early.
            if self == initial_edge:
                next.trace(edge_list, next)
                return
        else:
            edge_list.append(self)
        # As a breaking criterion, the initial edge of the vectorline
        # is passed down and, if the next edge in line would be the
        # initial edge again, we know that we are done and we stop the
        # recursion.
        if next != initial_edge:
            next.trace(edge_list, initial_edge)

    @abstractmethod
    def integrate(self):
        """Returns the oriented area covered by going from the start to
        the end of the line with respect to the origin.
        """
        pass

    @abstractmethod
    def split(self):
        """Splits the line according to the split parameters within
        self.splits at runtime.
        """
        pass

    @abstractmethod
    def check_within_bounds(self, param):
        """Checks if the parameter for the calculated crossing solution
        is within the uncritical range of the respective line.
        """
        pass

    @abstractmethod
    def get_midpoint(self):
        """Returns the midpoint of an Edge (giving its coordinates in
        the world system as a tuple).
        """
        pass

    @abstractmethod
    def get_gradient(self, point):
        """Note: All gradients are not normalized by default and are
        pointing to the right of the direction of the edge, meaning
        they point to the outside in the context of additive edges and
        to the inside in the context of subtractive edges.
        """
        pass

    @abstractmethod
    def get_distance(self, point):
        """This function serves to enable the VectorLines'
        'check_inside' method.  It returns the euclidean distance of a
        given point from the edge (if one is defined - only distance
        vectors orthogonal to the edge are deemed valid) as well as a
        boolean value that is True if the point is to the left of the
        edge (meaning 'on the inside' for additive objects) and False
        if it is to the right.
        """
        pass

    @abstractmethod
    def draw_edge(self):
        pass


class Straight(Edge):
    """Class for straight line pieces.  These are defined by their
    start and end points.
    """

    def __init__(self, source, startpoint, endpoint):
        # Call super.
        super(Straight, self).__init__(source, startpoint, endpoint)
        # Register start/end vectors (tuples of x and y value).
        self.direction = (endpoint.coords[0] - startpoint.coords[0],
                          endpoint.coords[1] - startpoint.coords[1])

    def _get_crossings(self):
        # Create an empty transitions list.
        tlist = []
        # Switch cases depending on the line going forward or
        # backward with respect to x.
        if self.startpoint.coords[0] < self.endpoint.coords[0]:
            start = self.startpoint.coords
            stop = self.endpoint.coords
            reversed = False
        else:
            start = self.endpoint.coords
            stop = self.startpoint.coords
            reversed = True
        # Calculate the slope (with respect to both axes, for ease
        # of computation).
        delta_x = stop[0] - start[0]
        delta_y = stop[1] - start[1]
        if delta_x != 0.0:
            m_x = delta_y / delta_x
        if delta_y != 0.0:
            m_y = delta_x / delta_y
        """
        m_x = (stop[1] - start[1]) / (stop[0] - start[0])
        m_y = 1.0 / m_x
        """
        # Define the coordinate ranges with respect to x...
        xrange = range(int(round(start[0])), int(round(stop[0])))
        # ... and y.  Another case switch is necessary depending on
        # the line pointing upward or downward.
        if start[1] < stop[1]:
            yrange = range(int(round(start[1])), int(round(stop[1])))
        else:
            yrange = range(int(round(start[1])) - 1,
                           int(round(stop[1])) - 1,
                           -1)

        # Initialize variables
        last = (start[0] - 1, start[1])
        x_iter = iter(xrange)
        y_iter = iter(yrange)
        x_next = None
        y_next = None

        while True:
            # Refill x_next if necessary.
            if x_next is None:
                try:
                    # Pull the value in x from the iterator and
                    # evaluate for y by using the slope defined
                    # above.
                    p = next(x_iter) + 0.5
                    x_next = (p, m_x * (p - start[0]) + start[1])
                except StopIteration:
                    # If we land here, there is no more value to
                    # pull.  This might mean that the loop can be
                    # broken, but only if we have also seen all
                    # values in y_iter.
                    # The following if contains a little hack to
                    # avoid an error when both x_next and y_next
                    # are originally uninitialized when the while
                    # loop is at first entered.
                    if y_next and y_next == stop:
                        break
                    x_next = stop
            # Refill y_next if necessary.
            if y_next is None:
                try:
                    # Pull the value in y from the iterator and
                    # evaluate for x by using the slope defined
                    # above.
                    p = next(y_iter) + 0.5
                    y_next = (m_y * (p - start[1]) + start[0], p)
                except StopIteration:
                    # If we land here, there is no more value to
                    # pull.  This might mean that the loop can be
                    # broken, but only if we have also seen all
                    # values in x_iter.
                    # Due to the order of execution, the same hack
                    # as above is not necessary here.
                    if x_next == stop:
                        break
                    y_next = stop
            # Switch cases depending on which point goes next.
            if x_next[0] < y_next[0]:
                # This if-clause takes care of discarding pseudo
                # duplicates.
                if (x_next[0] - last[0] > 1e-4
                        or abs(x_next[1] - last[1]) > 1e-4):
                    tlist.append(x_next)
                    last = x_next
                else:
                    last = (round(last[0], 1), round(last[1], 1))
                    tlist[-1] = last
                # Delete x_next so it can be refilled when looping
                # back.
                x_next = None
            else:
                # This if-clause takes care of discarding pseudo
                # duplicates.
                if (y_next[0] - last[0] > 1e-5
                        or abs(y_next[1] - last[1]) > 1e-4):
                    tlist.append(y_next)
                    last = y_next
                else:
                    last = (round(last[0], 1), round(last[1], 1))
                    tlist[-1] = last
                # Delete x_next so it can be refilled when looping
                # back.
                y_next = None
            # End of while-loop
        # Reverse if that is necessary (as found out at the start)
        if reversed:
            tlist.reverse()
        return tlist

    def check_within_bounds(self, param):
        legal = bool(-1e-6 < param < 1.0 + 1e-6)
        critical = bool(abs(param) < 1e-6 or abs(param - 1.0) < 1e-6)
        return legal, critical

    def integrate(self):
        return _integrate_straight_line(self.startpoint.coords,
                                        self.endpoint.coords)

    def split(self):
        if not self.splits:
            return False
        self.splits.sort(key=lambda x: x[0])
        prev_edge = Straight(self.source,
                             self.startpoint,
                             self.splits[0][1])
        self.startpoint.replace_edge(self, prev_edge)
        for split_index in range(len(self.splits)):
            current_node = self.splits[split_index][1]
            try:
                next_node = self.splits[split_index + 1][1]
            except IndexError:
                next_node = self.endpoint
            current_edge = Straight(self.source,
                                    current_node,
                                    next_node)
            current_node.register_edges(prev_edge, current_edge)
            prev_edge = current_edge
        self.endpoint.replace_edge(self, current_edge)
        return True

    def calc_split_point(self, split):
        return (self.startpoint.coords[0] + split * self.direction[0],
                self.startpoint.coords[1] + split * self.direction[1])

    def get_midpoint(self):
        return self.calc_split_point(0.5)

    def get_gradient(self, point=None):
        """Note: All gradients are not normalized by default and are
        pointing to the right of the direction of the edge, meaning
        they point to the outside in the context of additive edges and
        to the inside in the context of subtractive edges.
        """
        return (self.direction[1], -self.direction[0])

    def get_distance(self, point):
        """This function serves to enable the VectorLines'
        'check_inside' method.  It returns the euclidean distance of a
        given point from the edge (if one is defined - only distance
        vectors orthogonal to the edge are deemed valid) as well as a
        boolean value that is True if the point is to the left of the
        edge (meaning 'on the inside' for additive objects) and False
        if it is to the right.
        """
        grad = self.get_gradient()
        grad_len = math.sqrt(grad[0] ** 2 + grad[1] ** 2)
        grad = (grad[0] / grad_len, grad[1] / grad_len)
        params = _straight_cross(sp1=self.startpoint.coords,
                                 dir1=self.direction,
                                 sp2=point,
                                 dir2=grad)
        legal, _ = self.check_within_bounds(params[0])
        if legal:
            dist = -params[1]
        else:
            # The distance is set to inf if the base point for the
            # distance is not within the bounds of the edge.
            dist = float('inf')
        return dist

    def get_point_on_straight(self, point):
        """This function tells its caller whether or not a given point
        is on the line (with some leeway for numerical error) and, if
        so, what parameter along the line is associated with it.
        """
        if (point == self.startpoint.coords
                or point == self.endpoint.coords):
            return None
        grad = self.get_gradient()
        grad_len = math.sqrt(grad[0] ** 2 + grad[1] ** 2)
        grad = (grad[0] / grad_len, grad[1] / grad_len)
        params = _straight_cross(sp1=self.startpoint.coords,
                                 dir1=self.direction,
                                 sp2=point,
                                 dir2=grad)
        if abs(params[1]) < 1e-10 and bool(0.0 < params[0] < 1.0):
            return params[0]
        else:
            return None

    def draw_edge(self, field_proto, pass_down=None):
        # This function needs to do very different things if it is
        # called on the first edge of an object for the second time.
        # This is the case that is first dealt with in the following.
        if self.first_to_draw:
            # If the firstpoint attibute got a real point assigned, we
            # can complete the piece that got passed down to this
            # method and we are done.
            px = int(round(self.startpoint.coords[0]))
            py = int(round(self.startpoint.coords[1]))
            if self.firstpoint:
                area = (pass_down[1]
                        + _integrate_straight_line(self.startpoint.coords,
                                                   self.firstpoint))
                field_proto[px, py].append((pass_down[0], self.firstpoint, area))
            else:
                # If that was not the case, there are still two
                # variants remaining.
                # If the point that was passed down as the starting
                # point is the startpoint of this very edge, it means
                # there are no pixel border crossings at all on the
                # current object's border. This means we can finish by
                # defining this as an internal area (signaled by having
                # None as both beginning and end points).
                if pass_down[0] == self.startpoint.coords:
                    field_proto[px, py].append((None, None, pass_down[1]))
                # If this is not the case, we can be sure that a point
                # was written to field_proto already, but one with an
                # illegal starting point (namely this edge's starting
                # point).  We need to find it again in the list and
                # replace it with a corrected area attribute and the
                # real starting point which was passed down to this
                # method.
                else:
                    for i in range(len(field_proto[px, py])):
                        if field_proto[px, py][i][0] == self.startpoint.coords:
                            area = pass_down[1] + field_proto[px, py][i][2]
                            field_proto[px, py].append(
                                (pass_down[0], field_proto[px, py][i][1], area)
                            )
                            field_proto[px, py].pop(i)
                            break
                    else:
                        raise NotImplementedError("Something went wrong. "
                                                  "Please examine.")
            return
        if not pass_down:
            self.first_to_draw = True
        # This is where the main work is done.
        # First, get the list of all transitions from pixel to pixel to
        # work through.
        tlist = self._get_crossings()
        # The following has some special and corner cases that need to
        # be dealt with separately.  The first is to use differrent
        # code paths if the transition list is empty.
        if len(tlist) > 0:
            # First, deal with the segment before the first transition.
            # Another special case: If this is the first edge that is
            # processed, do not to anything right away but only save
            # the coordinates of the first transition point for later
            # use (when this method is called again by the last edge of
            # the border.
            if self.first_to_draw:
                self.firstpoint = tlist[0]
            else:
                # If this edge is not the first of the object, we are
                # guaranteed to have some info passed down to it by the
                # edge before.  We need to identify the pixel, add this
                # edge's contribution to the area covered and write it
                # to field_proto.
                px = int(round(self.startpoint.coords[0]))
                py = int(round(self.startpoint.coords[1]))
                area = (pass_down[1]
                        + _integrate_straight_line(self.startpoint.coords,
                                                   tlist[0]))
                field_proto[px, py].append((pass_down[0], tlist[0], area))
            # The middle points between two transitions can be dealt
            # with in a generic way.  This happens in the for loop.
            for t in range(len(tlist) - 1):
                # An in-between point between two transitions is
                # calculated and then rounded to the nearest integer,
                # which gives us the next pixel to deal with.
                px = int(round(0.5 * (tlist[t][0] + tlist[t + 1][0])))
                py = int(round(0.5 * (tlist[t][1] + tlist[t + 1][1])))
                area = _integrate_straight_line(tlist[t], tlist[t + 1])
                # Write the start point, end point and oriented area
                # contribution to the field_proto array based on which
                # the actual drawing will be done later down the line.
                field_proto[px, py].append((tlist[t], tlist[t + 1], area))
            # When the last transition is reached, the current edge
            # prepares data in order for the next in line to finish
            # (see the code above for that).
            area_part = _integrate_straight_line(tlist[-1],
                                                 self.endpoint.coords)
            segment_start = tlist[-1]
        else:
            # We land here if the list of transitions is empty. In this
            # case we first calculate the pixel we need to consider and
            # the area contribution of this edge.
            px = int(round(self.startpoint.coords[0]))
            py = int(round(self.startpoint.coords[1]))
            area = _integrate_straight_line(self.startpoint.coords,
                                            self.endpoint.coords)
            # If, additionally, this is the first edge piece
            # considered, we use the startpoint of this edge piece as
            # the start point of the pixel segment.  This is quite
            # hackish, as this will not yield a valid entry in
            # field_proto when one of the following edges write to it,
            # but it enables us to identify and deal with this case
            # when this edge piece's draw_edge method is called for
            # the second time.
            if self.first_to_draw:
                area_part = area
                segment_start = self.startpoint.coords
            else:
                # If it is not the first edge, we add this edge's area
                # contribution and pass the result down further.
                area_part = pass_down[1] + area
                segment_start = pass_down[0]
        # Finally, new_pass_down is created.  Its structure is
        #  1. A list/tuple of two integers: The coordinates of the
        #     starting point of the edge piece.
        #  2. The area contribution calculated so far.
        new_pass_down = (segment_start, area_part)
        # Recursively call the next edge in line.  Recursion is broken
        # at the top of this method, when the first edge's draw_edge
        # method is called for the second time.
        self.endpoint.get_edge(self).draw_edge(field_proto, new_pass_down)
        return


class Arc(Edge):

    def __init__(self, source, begin, end, startpoint, endpoint):
        # Call super
        super(Arc, self).__init__(source, startpoint, endpoint)
        # Arcs are stored differently compared to straight lines in
        # that they permanently link back to the original ellipse shape
        # object for all parameters it needs.
        self.begin = begin
        self.end = end

    def _get_crossings(self):
        try:
            tlist = self.source.tlist
        except AttributeError:
            self.source._el_crossings()
            tlist = self.source.tlist
        if self.source.additive:
            lo, hi = self.begin, self.end
        else:
            lo, hi = self.end, self.begin
        for lower in range(len(tlist)):
            if tlist[lower][2] > lo:
                break
        else:
            lower = len(tlist)
        for upper in range(lower, len(tlist)):
            if tlist[upper][2] > hi:
                break
        else:
            upper = len(tlist)
        tlist_part = tlist[lower:upper]
        if not self.source.additive:
            tlist_part.reverse()
        return tlist_part

    def _theta2par(self, theta):
        return (theta - self.begin) / (self.end - self.begin)

    def _par2theta(self, param):
        return self.begin + param * (self.end - self.begin)

    def check_within_bounds(self, point, in_el_coords=False):
        """In the case of Arc, 'param' is expected to be a point on the
        ellipse, from which we calculate theta of that point.
        """
        if in_el_coords:
            transformed_vector = point
        else:
            transformed_vector = (
                self.transform(self.translate(point, True), True))
        cosine_pre = transformed_vector[0] / self.source.len_1
        cosine = max(min(cosine_pre, 1.0), -1.0)
        if abs(cosine - cosine_pre) > 1e-6:
            warnings.warn("Cosine value has been clipped to fit inside the "
                          "mathematical domain [-1,1] and %f has been cut "
                          "away. This might indicate errors in the algorithm."
                          % (cosine - cosine_pre))
        if transformed_vector[1] > 0.0:
            theta = math.acos(cosine)
        else:
            theta = -math.acos(cosine)
        if self.source.additive:
            legal = bool(self.begin - 1e-6 < theta < self.end + 1e-6)
        else:
            legal = bool(self.end - 1e-6 < theta < self.begin + 1e-6)
        critical = bool(abs(theta - self.begin) < 1e-6
                        or abs(theta - self.end) < 1e-6)
        return legal, critical, self._theta2par(theta)

    def integrate(self):
        return self.source._integrate(self.begin, self.end)

    def transform(self, vector, to_el_system):
        return self.source._transform(vector, to_el_system)

    def translate(self, vector, to_el_system):
        return self.source._translate(vector, to_el_system)

    def split(self):
        if not self.splits:
            return False
        self.splits.sort(key=lambda x: x[0])
        current_end = self._par2theta(self.splits[0][0])
        next_node = self.splits[0][1]
        prev_edge = Arc(self.source,
                        self.begin,
                        current_end,
                        self.startpoint,
                        next_node)
        self.startpoint.replace_edge(self, prev_edge)
        for split_index in range(len(self.splits)):
            current_begin = current_end
            current_node = next_node
            try:
                current_end = self._par2theta(self.splits[split_index + 1][0])
                next_node = self.splits[split_index + 1][1]
            except IndexError:
                current_end = self.end
                next_node = self.endpoint
            current_edge = Arc(self.source,
                               current_begin,
                               current_end,
                               current_node,
                               next_node)
            current_node.register_edges(prev_edge, current_edge)
            prev_edge = current_edge
        self.endpoint.replace_edge(self, current_edge)
        return True

    def get_midpoint(self):
        return self.source._get_point(0.5 * (self.begin + self.end))

    def get_gradient(self, point):
        """Note: All gradients are not normalized by default and are
        pointing to the right of the direction of the edge, meaning they
        point to the outside in the context of additive edges and to the
        inside in the context of subtractive edges.
        """
        grad = self.source._get_gradient(point)
        if self.begin > self.end:
            grad = (-grad[0], -grad[1])
        return grad

    def get_distance(self, point):
        """This function serves to enable the VectorLines'
        'check_inside' method.  It returns the euclidean distance of a
        given point from the edge (if one is defined - only distance
        vectors orthogonal to the edge are deemed valid) as well as a
        boolean value that is True if the point is to the left of the
        edge (meaning 'on the inside' for additive objects) and False if
        it is to the right.
        """
        ecs_point = self.transform(self.translate(point, True), True)
        rad1 = self.source.len_1
        rad2 = self.source.len_2
        params = ((rad1 ** 4 * rad2 ** 4 - ecs_point[0] ** 2 * rad1 ** 2 * rad2 ** 4
                   - ecs_point[1] ** 2 * rad1 ** 4 * rad2 ** 2),
                  2 * (rad1 ** 2 * rad2 ** 4 + rad1 ** 4 * rad2 ** 2
                       - (ecs_point[0] ** 2 + ecs_point[1] ** 2) * rad1 ** 2 * rad2 ** 2),
                  (4 * rad1 ** 2 * rad2 ** 2 + rad1 ** 4 + rad2 ** 4
                   - ecs_point[0] ** 2 * rad1 ** 2 - ecs_point[1] ** 2 * rad2 ** 2),
                  2 * (rad1 ** 2 + rad2 ** 2),
                  1)
        solutions = poly.polyroots(params)
        dist = float('inf')
        if (abs(solutions[0] - solutions[1]) < 1e-2
                and abs(solutions[1] - solutions[2]) < 1e-2
                and abs(solutions[2] - solutions[3]) < 1e-2):
            dist = -rad1
        else:
            for sol in solutions:
                if float(sol.imag) == 0.0:
                    ck = float(sol.real)
                    cpoint = (ecs_point[0] / (1.0 + (ck / rad1 ** 2)),
                              ecs_point[1] / (1.0 + (ck / rad2 ** 2)))
                    legal, _, _ = self.check_within_bounds(cpoint,
                                                           in_el_coords=True)
                    if legal:
                        cdist = math.sqrt((cpoint[0] - ecs_point[0]) ** 2
                                          + (cpoint[1] - ecs_point[1]) ** 2)
                        if cdist < abs(dist):
                            dist = math.copysign(cdist, ck)
            # end of for-loop over quartic solutions
        # Reminder: To satisfy the condition of returning True if the
        # reference point is on the right and False if it is on the
        # left of the edge, a case switch needs to happen here as, up
        # until now, the edge's direction has not been considered.
        if self.begin > self.end:
            dist = -dist
        return dist

    def trace(self, edge_list, initial_edge):
        """Traces the shape of a newly defined VectorLine object.
        Overrides the base trace function to perform some numerical
        error correction.
        """
        # Specifically for arcs, we frequently run into a problem where
        # the values for the first/last legal theta (internal
        # parameters 'begin'/'end') do not 100% line up with the
        # locations of the start/endpoints. This difference can lead to
        # multiple problems down the line (failing or giving slightly
        # wrong values when drawing), so we detect and close it here by
        # introducing a short piece of straight line.
        # CHANGED: The first implementation led to clutter and,
        # ultimately, less reliable code. This is why we now only
        # introduce new line pieces if we would miss a pixel border
        # crossing.  All other numerical inaccuracies will be dealt
        # with within the drawing algorithm, not introducing additional
        # edges but just calculating and a correction.
        # First, get the coordinates of the points that would fit
        # begin/end exactly.
        truesp = self.source._get_point(self.begin)
        trueep = self.source._get_point(self.end)
        # Test if the difference between them and the current
        # startpoint/endpoint is significant.  It is always significant
        # if it leads to a dislocation regarding the pixel raster;
        # otherwise it only is if the difference in coordinates is
        # above a reasonable threshold.
        if (round(truesp[0]) != round(self.startpoint.coords[0])
                or round(truesp[1]) != round(self.startpoint.coords[1])):
            # If a replacement for the startpoint seems in order, we
            # define and insert it here...
            startnode = Node(truesp)
            startstraight = Straight(self.source, self.startpoint, startnode)
            self.startpoint.replace_edge(self, startstraight)
            startnode.register_edges(startstraight, self)
            self.startpoint = startnode
            # ...and, once we are done, we call the trace function of
            # the freshly created line piece so it gets added to the
            # normal edge list of the calling VectorLine.  Note that
            # this will lead to this very function being called again,
            # but next time around we will not end up here anymore
            # because we established a perfect fit for this arc's
            # starting point.
            if self == initial_edge:
                startstraight.trace(edge_list, startstraight)
            else:
                startstraight.trace(edge_list, initial_edge)
        elif (round(trueep[0]) != round(self.endpoint.coords[0])
              or round(trueep[1]) != round(self.endpoint.coords[1])):
            # If a replacement for the endpoint seems in order, we
            # define and insert it here...
            endnode = Node(trueep)
            endstraight = Straight(self.source, endnode, self.endpoint)
            self.endpoint.replace_edge(self, endstraight)
            endnode.register_edges(self, endstraight)
            self.endpoint = endnode
            # ...and, once we are done, we call the edges' regular
            # trace function.  Since the new line piece is now properly
            # registered, its own trace function will be called next
            # without further action required.
            super(Arc, self).trace(edge_list, initial_edge)
        else:
            # If the points are not that far off, we continue by just
            # calling the edges' regular trace function.
            super(Arc, self).trace(edge_list, initial_edge)

    def draw_edge(self, field_proto, pass_down=None):
        # This function needs to do very different things if it is
        # called on the first edge of an object for the second time.
        # This is the case that is first dealt with in the following.
        if self.first_to_draw:
            # In all cases, however, we need to identify the pixel
            # first.  We can just use the edge's startpoint for that.
            px = int(round(self.startpoint.coords[0]))
            py = int(round(self.startpoint.coords[1]))
            if self.firstpoint:
                # If the firstpoint attibute got a real point assigned,
                # we can complete the piece that got passed down to
                # this method and we are done.
                area = (pass_down[1]
                        + (_integrate_straight_line(self.startpoint.coords,
                                                    self.source._get_point(self.begin)))
                        + self.source._integrate(self.begin,
                                                 self.firstpoint[2]))
                field_proto[px, py].append((pass_down[0],
                                            self.firstpoint[:2],
                                            area))
            else:
                # If that was not the case, there are still two
                # variants remaining.
                if pass_down[0] == self.startpoint.coords:
                    # If the point that was passed down as the starting
                    # point is the startpoint of this very edge, it
                    # means there are no pixel border crossings at all
                    # on the current object's border. This means we can
                    # finish by defining this as an internal area
                    # (signaled by having None as both beginning and
                    # end points).
                    field_proto[px, py].append((None, None, pass_down[1]))
                else:
                    # If this is not the case, we can be sure that a
                    # point was written to field_proto already, but one
                    # with an illegal starting point (namely this
                    # edge's starting point).  We need to find it again
                    # in the list and replace it with a corrected area
                    # attribute and the real starting point which was
                    # passed down to this method.
                    for i in range(len(field_proto[px, py])):
                        if field_proto[px, py][i][0] == self.startpoint.coords:
                            area = pass_down[1] + field_proto[px, py][i][2]
                            field_proto[px, py][i] = ((pass_down[0],
                                                       field_proto[px, py][i][1],
                                                       area))
                            break
                    else:
                        raise NotImplementedError("Something went wrong. "
                                                  "Please examine.")
            return
        if not pass_down:
            self.first_to_draw = True
        # This is where the main work is done.
        # First, get the list of all transitions from pixel to pixel to
        # work through.
        tlist = self._get_crossings()
        # The following has some special and corner cases that need to
        # be dealt with separately.  The first is to use differrent
        # code paths if the transition list is empty.
        if len(tlist) > 0:
            # First, deal with the segment before the first transition.
            # Another special case: If this is the first edge that is
            # processed, do not to anything right away but only save
            # the coordinates of the first transition point for later
            # use (when this method is called again by the last edge of
            # the border.
            if self.first_to_draw:
                self.firstpoint = tlist[0]
            else:
                # If this edge is not the first of the object, we are
                # guaranteed to have some info passed down to it by the
                # edge before.  We need to identify the pixel, add this
                # edge's contribution to the area covered and write it
                # to field_proto.
                px = int(round(self.startpoint.coords[0]))
                py = int(round(self.startpoint.coords[1]))
                # CHANGED: Despite our best efforts, it was noted that,
                # under numerically bad circumstances of two edges
                # intersecting under a very acute angle, there can be a
                # little gap between the endpoint of the first and the
                # startpoint of the second edge if at least one of them
                # is an arc.  While this is insignificant in terms of
                # length, due to the way the covered area of a pixel is
                # calculated, this can cause a significant numerical
                # error.  In order to counteract this, we include the
                # difference between the arc's logical (represented by
                # its 'startpoint' node) and numerical (represented by
                # the 'begin' attribute) starting point as a straight
                # line contributing to the total area covered, thus
                # making sure the vectorline is definitely closed.
                area = (pass_down[1]
                        + (_integrate_straight_line(self.startpoint.coords,
                                                    self.source._get_point(self.begin)))
                        + self.source._integrate(self.begin, tlist[0][2]))
                field_proto[px, py].append((pass_down[0], tlist[0][:2], area))
            # The middle points between two transitions can be dealt
            # with in a generic way.  This happens in the for loop.
            for t in range(len(tlist) - 1):
                # An in-between point between two transitions is
                # calculated and then rounded to the nearest integer,
                # which gives us the next pixel to deal with.
                pt = self.source._get_point(0.5 * (tlist[t][2] + tlist[t + 1][2]))
                px = int(round(pt[0]))
                py = int(round(pt[1]))
                area = self.source._integrate(tlist[t][2], tlist[t + 1][2])
                # Write the start point, end point and oriented area
                # contribution to the field_proto array.
                field_proto[px, py].append((tlist[t][:2], tlist[t + 1][:2], area))
            # When the last transition is reached, the current edge
            # prepares data in order for the next in line to finish
            # (see the code above for that).
            segment_start = tlist[-1][:2]
            area_part = (self.source._integrate(tlist[-1][2], self.end)
                         + (_integrate_straight_line(
                        self.source._get_point(self.end),
                        self.endpoint.coords)))
        else:
            # We land here if the list of transitions is empty. In this
            # case we first calculate the pixel we need to consider and
            # the area contribution of this edge.
            px = int(round(self.startpoint.coords[0]))
            py = int(round(self.startpoint.coords[1]))
            area = ((_integrate_straight_line(self.startpoint.coords,
                                              self.source._get_point(self.begin)))
                    + self.source._integrate(self.begin, self.end)
                    + (_integrate_straight_line(
                        self.source._get_point(self.end),
                        self.endpoint.coords)))
            # If, additionally, this is the first edge piece
            # considered, we use the startpoint of this edge piece as
            # the start point of the pixel segment.  This is quite
            # hackish, as this will not yield a valid entry in
            # field_proto when one of the following edges writes to it,
            # but it enables us to identify and deal with this case
            # when this edge piece's draw_edge method is called for
            # the second time.
            if self.first_to_draw:
                area_part = area
                segment_start = self.startpoint.coords
            else:
                # If it is not the first edge, we add this edge's area
                # contribution and pass the result down further.
                area_part = pass_down[1] + area
                segment_start = pass_down[0]
        # Finally, new_pass_down is created.  Its structure is
        #  1. A list/tuple of two integers: The coordinates of the
        #     starting point of the edge piece.
        #  2. The area contribution calculated so far.
        new_pass_down = (segment_start, area_part)
        # Recursively call the next edge in line.  Recursion is broken
        # at the top of this method, when the first edge's draw_edge
        # method is called for the second time.
        self.endpoint.get_edge(self).draw_edge(field_proto, new_pass_down)
        return


@six.add_metaclass(ABCMeta)
class Shape:
    """This is the base class for all shapes."""

    def __init__(self):
        self.fused = False
        self.edge_list = None

    def _set_geometry(self, pos_x, pos_y, len_1, len_2, angle):
        self.centerpoint = (pos_x, pos_y)
        self.len_1 = len_1
        self.len_2 = len_2
        self.angle = angle
        self.sine = math.sin(angle)
        self.cosine = math.cos(angle)

    def get_shape_data(self):
        data_list = (self.__class__, self.centerpoint[0], self.centerpoint[1],
                     self.len_1, self.len_2, self.angle, self.additive)
        return data_list

    def split(self):
        for edge in self.edge_list:
            splitted = edge.split()
            if splitted:
                self.fused = True

    def draw(self, field_proto):
        # draw_edge of the first edge makes it draw itself and, to
        # simplify the flow, tell its following edge to draw itself
        # recursively.
        self.edge_list[0].draw_edge(field_proto)

    @abstractmethod
    def check_inside(self, point):
        pass


class Rectangle(Shape):
    """This implements a rectangle as a base shape."""

    def __init__(self, pos_x, pos_y, len_1, len_2,
                 angle, additive):
        """Changed with nnlib_multiobject_v06: The initializer needs to
        deal with 3D inputs from now on, but since there is no plan to
        make use of this here, we just quietly ignore them.
        """
        # remove arguments pos_z, len_3, beta, gamma #
        super(Rectangle, self).__init__()
        self._set_geometry(pos_x, pos_y, len_1, len_2, angle)
        self.area = 4.0 * len_1 * len_2
        self.additive = additive
        corner_0 = Node((pos_x + len_1 * self.cosine - len_2 * self.sine,
                         pos_y + len_1 * self.sine + len_2 * self.cosine))
        corner_1 = Node((pos_x - len_1 * self.cosine - len_2 * self.sine,
                         pos_y - len_1 * self.sine + len_2 * self.cosine))
        corner_2 = Node((pos_x - len_1 * self.cosine + len_2 * self.sine,
                         pos_y - len_1 * self.sine - len_2 * self.cosine))
        corner_3 = Node((pos_x + len_1 * self.cosine + len_2 * self.sine,
                         pos_y + len_1 * self.sine - len_2 * self.cosine))
        # If the rectangle should be subtracted, the order of corners
        # needs to be changed. Luckily, switching out corners 1 and 3
        # does everything necessary so the rest of the code does not
        # need any branches.
        if not additive:
            corner_1, corner_3 = corner_3, corner_1

        self.edge_list = (Straight(self, corner_0, corner_1),
                          Straight(self, corner_1, corner_2),
                          Straight(self, corner_2, corner_3),
                          Straight(self, corner_3, corner_0))
        corner_0.register_edges(self.edge_list[3], self.edge_list[0])
        corner_1.register_edges(self.edge_list[0], self.edge_list[1])
        corner_2.register_edges(self.edge_list[1], self.edge_list[2])
        corner_3.register_edges(self.edge_list[2], self.edge_list[3])

    def check_inside(self, point):
        x_trans = point[0] - self.centerpoint[0]
        y_trans = point[1] - self.centerpoint[1]
        if (abs(x_trans * self.cosine + y_trans * self.sine) <= self.len_1
                and abs(-x_trans * self.sine + y_trans * self.cosine) <= self.len_2):
            return True
        else:
            return False


class Ellipse(Shape):
    """This implements an ellipse as a base shape."""

    def __init__(self, pos_x, pos_y,
                 len_1, len_2,
                 angle, additive):
        """Changed with nnlib_multiobject_v06: The initializer needs to
        deal with 3D inputs from now on, but since there is no plan to
        make use of this here, we just quietly ignore them.
        """
        super(Ellipse, self).__init__()
        self._set_geometry(pos_x, pos_y, len_1, len_2, angle)
        self._create_implicit()
        self.area = math.pi * len_1 * len_2
        self.additive = additive
        corner = Node((pos_x - len_1 * self.cosine, pos_y - len_1 * self.sine))
        if additive:
            self.edge_list = [Arc(self, -math.pi, math.pi, corner, corner)]
        else:
            self.edge_list = [Arc(self, math.pi, -math.pi, corner, corner)]
        corner.register_edges(self.edge_list[0], self.edge_list[0])

    def _integrate(self, begin, end):
        """Performs the integration on ellipse slices.  The formula
        employed here stems from symbolic integration of the explicit
        representation of the ellipse parametrized over the angle theta
        and is therefore exact save for numerical inaccuracies."""
        a = self.len_1 * self.cosine
        b = self.len_2 * self.sine
        c = self.len_1 * self.sine
        d = self.len_2 * self.cosine
        adbc = a * d + b * c
        cxay = c * self.centerpoint[0] - a * self.centerpoint[1]
        dxby = d * self.centerpoint[0] + b * self.centerpoint[1]
        return (adbc * (end - begin) + cxay * (math.cos(end) - math.cos(begin))
                + dxby * (math.sin(end) - math.sin(begin)))

    def _el_crossings(self):
        """Returns a list of every crossing of the ellipse's border
        from one pixel to another.
        """

        def _cross(current, params, center):
            """Performs the actual calculations of theta and x (or y)
            depending on the given y (or x) coordinate in 'current'.
            'params' and 'center' have to be prepared in such a way
            that they fit the necessary steps.  That is, if x is to be
            calculated, the order should be (a,b,c,d) and (x_center,
            y_center) because the algorithm was written with respect to
            this case.  For calculations of y, the order is (c,-d,a,-b)
            and (y_center,x_center).
            """
            # Initialize empty lists for the values to be returned.
            theta = []
            calculated = []
            try:
                # We employ a two-step process. First, calculate the
                # square root within the expression.  This may fail due
                # to negative values under the root.  In this case, we
                # encounter a value error that is dealt with below.
                radix = math.sqrt(params[0] ** 2
                                  + params[1] ** 2
                                  - (current - center[0]) ** 2)
                # Second, calculate the angle.  This may fail due to a
                # zero division, which is dealt with below as well.
                theta.append(
                    2 * math.atan((radix - params[1])
                                  / (params[0] + current - center[0]))
                )
                if radix > 0.0:
                    # If the root turns out to be exactly zero, there
                    # is no reason to return the same solution twice.
                    # Therefore, the second value is only attached if
                    # that is not the case.
                    theta.append(
                        2 * math.atan((-radix - params[1])
                                      / (params[0] + current - center[0]))
                    )
            except ZeroDivisionError:
                # In case of a zero division, we have to consider a
                # special case.  Pi is always a solution, then, and a
                # second one exists if the second parameter (b or -d,
                # respectively) is nonzero.
                theta.append(math.pi)
                if params[1] != 0:
                    theta.append(2 * math.atan(params[0] / params[1]))
            except ValueError:
                # If we land here, the square root turns out to be of a
                # negative number, meaning we are beyond the ellipse.
                # No real solution exists, then, and empty lists for
                # solutions will be returned.  It is the caller's job
                # to handle the situation accordingly.
                pass

            for t in theta:
                # Calculate the other respective coordinate.
                calculated.append(params[2] * math.cos(t)
                                  + params[3] * math.sin(t)
                                  + center[1])
            return theta, calculated

        a = self.len_1 * self.cosine
        b = self.len_2 * self.sine
        c = self.len_1 * self.sine
        d = self.len_2 * self.cosine

        # The mathematical expression for calculating theta under given
        # x or y are similar enough to reuse the same code for both,
        # but some of the inputs need to be reordered.  The
        # implementation in '_cross' has been done with respect to x,
        # so the inputs can be taken directly for that case, but need
        # to be changed around for y.
        params_x = (c, -d, a, -b)
        center_x = (self.centerpoint[1], self.centerpoint[0])
        params_y = (a, b, c, d)
        center_y = (self.centerpoint[0], self.centerpoint[1])

        # Initialize the list.
        tlist = []

        # The main part of the code.  The following iterates over the
        # ellipse exactly four times: Negative and positive x and y
        # direction from the center.
        # The code has been refactored to harness the fact that all
        # crossings already come up in the right order if they are
        # written to separate lists and reversed as necessary, so no
        # extra ordering step, hampering performance, is necessary.

        # Initialize the intermediate list for the left and right
        # crossings.
        tlist_up = []
        tlist_down = []
        # Start at the borders down from the center point.
        current = round(self.centerpoint[1]) - 0.5
        # Loop over the lower part of the ellipse.  See below for
        # breaking conditions.
        while True:
            # '_cross' returns the thetas and the x coordinates.
            theta, calculated = _cross(current, params_x, center_x)
            if len(theta) == 2:
                # Save the solutions into the left and right list,
                # depending on the returned solutions for x.
                if calculated[0] > calculated[1]:
                    tlist_up.append((calculated[0], current, theta[0]))
                    tlist_down.append((calculated[1], current, theta[1]))
                else:
                    tlist_up.append((calculated[1], current, theta[1]))
                    tlist_down.append((calculated[0], current, theta[0]))
            # If only a single or no solution has been returned, the
            # loop can be broken.
            elif len(theta) == 1:
                tlist_down.append((calculated[0], current, theta[0]))
                break
            else:
                break
            # Decrement the 'current' pointer.
            current -= 1.0
        # Both lists now need to be reversed in place to fit when we
        # loop over the upper half in the following.
        tlist_up.reverse()
        tlist_down.reverse()
        # Re-initialize 'current' to the border just above the center
        # point.
        current = round(self.centerpoint[1]) + 0.5
        # The loop carries out in exactly the same way as the one
        # above.
        while True:
            theta, calculated = _cross(current, params_x, center_x)
            if len(theta) == 2:
                if calculated[0] > calculated[1]:
                    tlist_up.append((calculated[0], current, theta[0]))
                    tlist_down.append((calculated[1], current, theta[1]))
                else:
                    tlist_up.append((calculated[1], current, theta[1]))
                    tlist_down.append((calculated[0], current, theta[0]))
            elif len(theta) == 1:
                tlist_down.append((calculated[0], current, theta[0]))
                break
            else:
                break
            current += 1.0
        # The (now complete) list of crossings on the left is reversed
        # again...
        tlist_down.reverse()
        # ...to fit the list of crossings on the right, such that they
        # are ordered in mathematically positive circular direction
        # when combined.
        tlist_x = tlist_up + tlist_down
        # In the finished list, find the point where theta rolls over
        # from pi to -pi and rearrange it such that this rollover point
        # in the midst of the list is eliminated.
        for i in range(len(tlist_x) - 1):
            if tlist_x[i][2] > tlist_x[i + 1][2]:
                tlist_x = tlist_x[i + 1:] + tlist_x[:i + 1]
                break
        # Re-initialize the intermediate lists for crossing detection
        # in x direction.
        tlist_up = []
        tlist_down = []
        # Reset current to the border just left of the center point.
        current = round(self.centerpoint[0]) - 0.5
        # The following loop is largely still the same as before, with
        # the exception of different parameter sets now being passed to
        # '_cross' and '_fill' being called to fill the inner space of
        # the ellipse.
        while True:
            theta, calculated = _cross(current, params_y, center_y)
            if len(theta) == 2:
                if calculated[0] > calculated[1]:
                    tlist_up.append((current, calculated[0], theta[0]))
                    tlist_down.append((current, calculated[1], theta[1]))
                else:
                    tlist_up.append((current, calculated[1], theta[1]))
                    tlist_down.append((current, calculated[0], theta[0]))
            elif len(theta) == 1:
                tlist_down.append((current, calculated[0], theta[0]))
                break
            else:
                break
            current -= 1.0
        # Reversing is the same in x and y direction.
        tlist_up.reverse()
        tlist_down.reverse()
        # Reset current again, this time to the immediate right border
        # of the center point.
        current = round(self.centerpoint[0]) + 0.5
        # While loop still basically stays the same.
        while True:
            theta, calculated = _cross(current, params_y, center_y)
            if len(theta) == 2:
                if calculated[0] > calculated[1]:
                    tlist_up.append((current, calculated[0], theta[0]))
                    tlist_down.append((current, calculated[1], theta[1]))
                else:
                    tlist_up.append((current, calculated[1], theta[1]))
                    tlist_down.append((current, calculated[0], theta[0]))
            elif len(theta) == 1:
                tlist_down.append((current, calculated[0], theta[0]))
                break
            else:
                break
            current += 1.0
        # Reverse the list of the upper border parts...
        tlist_up.reverse()
        # ...to again combine both lists into one that is already
        # implicitly sorted for the mathematically positive circular
        # direction.
        tlist_y = tlist_up + tlist_down
        # Again, the list gets rearranged such that it starts with
        # the nearest point to theta=-pi and ends with the nearest
        # point to theta=pi.
        for i in range(len(tlist_y) - 1):
            if tlist_y[i][2] > tlist_y[i + 1][2]:
                tlist_y = tlist_y[i + 1:] + tlist_y[:i + 1]
                break

        # Last, both lists need to be brought together for the final
        # result.  Sorting can once again be avoided by exploiting the
        # structure of the separate lists: Both start with their
        # respective closest point to theta = -pi and end with the
        # closest point to theta = pi, so we can keep their sorting and
        # use a zipping algorithm to unify them.

        # First, get an iterator out of each list.
        x_iter = iter(tlist_x)
        y_iter = iter(tlist_y)
        # Initialize some stuff: 'tlist' will be the resulting combined
        # list, 'last' will store the point that was last inserted into
        # 'tlist' (it is being initialized with a placeholder value to
        # avoid special cases in the following algorithm), 'x_next' and
        # 'y_next' store the respective next candidate point out of
        # each separate list.
        tlist = []
        last = (0.0, 0.0, -4.0)
        x_next = None
        y_next = None
        # Another while loop (with break rules that are specified and
        # checked within) performs the unification.
        while True:
            # First, check if x_next and/or y_next need to be
            # (re)filled.
            if x_next is None:
                try:
                    # Try to pull the next coordinate set from x_iter.
                    x_next = next(x_iter)
                except StopIteration:
                    # Should this fail with a StopIteration, we reached
                    # the end of x_iter.  Fill x_next with a dummy
                    # variable that will never occur in the real data
                    # or, if y_next already contains this same dummy
                    # variable, break the loop as we have seen all
                    # parameter tuples.
                    # The following if-clause contains a little hack
                    # (first checking that x_next is not None before
                    # comparing) so we avoid an error.  The same
                    # workaround is not required for y_next below.
                    if y_next and y_next == (0.0, 0.0, 4.0):
                        break
                    x_next = (0.0, 0.0, 4.0)
            # The same does essentially happen for y_next.
            if y_next is None:
                try:
                    y_next = next(y_iter)
                except StopIteration:
                    if x_next == (0.0, 0.0, 4.0):
                        break
                    y_next = (0.0, 0.0, 4.0)
            # Since both lists are already sorted, there are only
            # exactly two points that must be considered for the next
            # entry into 'tlist', and there is no case where none of
            # them must go next, so it is a simple if-else-decision.
            if x_next[2] > y_next[2]:
                # With theta of x_next being greater than y_next,
                # y_next is the next one to be entered.  Before doing
                # so, check if it is a pseudo duplicate of the last
                # entry (i.e. a point identical or very close to the
                # last one).  If such a point is found, it is discarded
                # as these are prone to throw the algorithm in
                # 'define_shape' off the rails and provide no vital
                # information anyway.
                if y_next[2] - last[2] > 1e-6:
                    tlist.append(y_next)
                    last = y_next
                else:
                    last = (round(last[0], 1), round(last[1], 1), last[2])
                    tlist[-1] = last
                # Empty y_next so it can be refilled in the next loop
                # run.
                y_next = None
            else:
                # In the 'else' case, x_next comes next; all other
                # considerations from above apply in the same way.
                if x_next[2] - last[2] > 1e-6:
                    tlist.append(x_next)
                    last = x_next
                else:
                    last = (round(last[0], 1), round(last[1], 1), last[2])
                    tlist[-1] = last
                x_next = None
        self.tlist = tlist

    def _create_implicit(self):
        """Calculates implicit form of the ellipse equation.  Called by
        __init__.  Implicit representation can be requested from the
        outside via fetch_implicit.
        The resulting parameter list must be interpreted as
          (params[0] + params[1]*x + params[2]*y
           + params[3]*x**2 + params[4]*x*y + y**2) = 0
        """
        xc, yc = self.centerpoint
        norm_factor = (self.sine ** 2 * self.len_2 ** 2
                       + self.cosine ** 2 * self.len_1 ** 2)
        self._impl = ((self.len_2 ** 2 * (xc * self.cosine + yc * self.sine) ** 2
                       + self.len_1 ** 2 * (xc * self.sine - yc * self.cosine) ** 2
                       - self.len_1 ** 2 * self.len_2 ** 2)
                      / norm_factor,
                      (-2.0 * self.len_2 ** 2 * self.cosine * (xc * self.cosine
                                                               + yc * self.sine)
                       + 2.0 * self.len_1 ** 2 * self.sine * (-xc * self.sine
                                                              + yc * self.cosine))
                      / norm_factor,
                      (-2.0 * self.len_2 ** 2 * self.sine * (xc * self.cosine
                                                             + yc * self.sine)
                       + 2.0 * self.len_1 ** 2 * self.cosine * (xc * self.sine
                                                                - yc * self.cosine))
                      / norm_factor,
                      (self.len_2 ** 2 * self.cosine ** 2
                       + self.len_1 ** 2 * self.sine ** 2)
                      / norm_factor,
                      2.0 * self.sine * self.cosine
                      * (self.len_2 ** 2 - self.len_1 ** 2) / norm_factor)

    def _transform(self, vector, to_el_system):
        """Performs a linear transformation on 'vector' from the
        cartesian coordinate system to the ellipse-centric coordinate
        system (both orthonormal) or back, depending on the truth value
        of 'to_el_system'.  Combine with '_translate' to get an affine
        transformation from one system to the other.  ('_transform'
        turns, '_translate' shifts the given vector)
        """
        if to_el_system:
            return (self.cosine * vector[0] + self.sine * vector[1],
                    -self.sine * vector[0] + self.cosine * vector[1])
        else:
            return (self.cosine * vector[0] - self.sine * vector[1],
                    self.sine * vector[0] + self.cosine * vector[1])

    def _translate(self, vector, to_el_system):
        """Performs an affine translation on 'vector' from the
        cartesian coordinate system to the ellipse-centric coordinate
        system or back, depending on the truth value of 'to_el_system'.
        Combine with '_transform' to get an affine transformation from
        one system to the other.  ('_transform' turns, '_translate'
        shifts the given vector)
        """
        if to_el_system:
            return (vector[0] - self.centerpoint[0],
                    vector[1] - self.centerpoint[1])
        else:
            return (vector[0] + self.centerpoint[0],
                    vector[1] + self.centerpoint[1])

    def _get_point(self, theta):
        """Returns cartesian coordinates of a point on the ellipse's
        border given an angle theta.
        """
        return ((self.len_1 * self.cosine * math.cos(theta)
                 - self.len_2 * self.sine * math.sin(theta)
                 + self.centerpoint[0]),
                (self.len_1 * self.sine * math.cos(theta)
                 + self.len_2 * self.cosine * math.sin(theta)
                 + self.centerpoint[1]))

    def _get_gradient(self, coords):
        """Returns the gradient of the implicit equation of the
        ellipse.  The result is an outward-pointing vector in cartesian
        coordinates as long as the point described by 'coords' does
        indeed lie on the border (this condition does not get checked,
        however!)
        """
        params = self._impl
        # Notice: Since we nowhere depend upon the gradient having the
        # specific length of its mathematical definition, we scale it
        # back.  This accelerates the convergence of the centerpoint
        # finding algorithm slightly.
        grad = (0.375 * (params[1] + 2 * params[3] * coords[0] + params[4] * coords[1]),
                0.375 * (params[2] + params[4] * coords[0] + 2 * coords[1]))
        return grad

    def fetch_implicit(self):
        """Returns the ellipse's representation as an implicit function.
        The parameter vector is to be interpreted in the following way:
          (params[0] + params[1]*x + params[2]*y
           + params[3]*x**2 + params[4]*x*y + y**2) = 0
        CHANGED in mo_renderer_new_v5b: The representation itself is
        calculated at init time by _create_implicit and saved as _impl,
        so this function only needs to pass it on.
        """
        return self._impl

    def check_inside(self, point):
        """Check if the given point is inside the ellipse (return True)
        or not (return False).  Implemented using the implicit
        representation of the ellipse.
        """
        params = self._impl
        if (params[0] + point[0] * params[1] + point[1] * params[2]
                + point[0] ** 2 * params[3] + point[0] * point[1] * params[4]
                + point[1] ** 2 <= 0.0):
            return True
        else:
            return False


class VectorLine(Shape):

    def __init__(self):
        super(VectorLine, self).__init__()

    def check_inside(self, point):
        """Checks whether a given point is on the inside of a vector
        line (return True) or not (return False).  It does that by
        going through all corners and edges of the shape, for each of
        which the shortest legal signed distance to the point is
        calculated.  After having gone through all corners and edges,
        the sign of the shortest overall distance is used as the
        indication of the point living outside or inside the shape.
        """
        distance = float('inf')
        for edge in self.edge_list:
            cdist = edge.get_distance(point)
            if abs(cdist) < abs(distance):
                distance = cdist
            cdist = edge.endpoint.get_distance(point, edge)
            if abs(cdist) < abs(distance):
                distance = cdist
        return (distance < 0.0) == self.additive

    def trace(self, node, start_index):
        """'trace' is called to get the list of edges that belong to
        the vector line and to calculate some essential information
        about its geometry, such as an inner point (called
        'centerpoint' for compatibility - it is generally not the
        shape's center in any geometrically meaningful way), the total
        area covered and whether or not it is overall an additive or a
        subtractive shape.
        """
        # Get initial edge list ready
        initial_edge = node.get_edge(start_index)
        self.edge_list = []
        # CHANGED in mo_renderer_new_v7: Sanitization (clearing out
        # edges of negligeable length) is done right away while
        # tracing.
        # Note on the 'trace' functions: After code refactoring,
        # 'trace' does not return the list of edges anymore.  Instead,
        # the list given as its first argument continuously gets
        # altered throughout the recursion, but not returned.  This
        # specifically means that, after 'trace' completes,
        # 'self.edge_list' will contain the expected list without any
        # reassignment.
        initial_edge.trace(self.edge_list, initial_edge)
        # Get the encompassed area
        area = 0.0
        for edge in self.edge_list:
            area += edge.integrate()
        # CHANGED: If the area of a new Vectorline is (close to) zero,
        # we dismiss it for only consisting of borders but no inner
        # area, which is not what we are interested in.
        if abs(area) < 1e-4:
            # Return False to the caller to signal that this vectorline
            # is not to be kept around.
            return False
        elif area > 0.0:
            self.additive = True
        else:
            self.additive = False
        # Note that, by the definition of the Leibnitz rule, the sum of
        # the integrals is double the area covered, so we multiply by
        # 0.5 to correct for this.
        self.area = abs(0.5 * area)
        coords = self.edge_list[0].get_midpoint()
        to_inside = self.edge_list[0].get_gradient(coords)
        if self.additive:
            to_inside = (-to_inside[0], -to_inside[1])
        self.centerpoint = (coords[0] + to_inside[0], coords[1] + to_inside[1])
        while not self.check_inside(self.centerpoint):
            to_inside = (0.5 * to_inside[0], 0.5 * to_inside[1])
            self.centerpoint = (coords[0] + to_inside[0], coords[1] + to_inside[1])
        return True


class FieldScope:
    class FieldProto:
        """To reduce the excessive amount of lists that are created and
        largely never used when drawing with the new renderer, this class
        (whose child objects are subscriptable) acts like the formerly used
        numpy array of lists, but it produces the lists on demand as
        necessary with no overhead.
        It has two modes of operation: When open (as it is when newly
        initialized), lists are dynamically created when a coordinate is
        requested that was not used before. When closed (close it via its
        'close' method), no new lists are created; instead, NoneTypes are
        returned when a list that does not exist is requested.
        Additionally, it can provide iterators over rows and columns (given
        a row) that only provide coordinates actually in use and the room
        between, again saving overhead.
        """

        def __init__(self):
            self.rowdict = {}
            self.closed = False

        def __getitem__(self, subscript):
            if len(subscript) != 2:
                raise TypeError("FieldProto objects are only subscriptable "
                                "with exactly 2 indices.")
            try:
                coldict = self.rowdict[subscript[0]]
            except KeyError:
                if self.closed:
                    return None
                else:
                    coldict = {}
                    self.rowdict[subscript[0]] = coldict
            try:
                lst = coldict[subscript[1]]
            except KeyError:
                if self.closed:
                    return None
                else:
                    lst = []
                    coldict[subscript[1]] = lst
            return lst

        def close(self):
            self.closed = True

        def row_iter(self):
            try:
                return range(min(self.rowdict), max(self.rowdict) + 1)
            except ValueError:
                return range(0)

        def col_iter(self, row):
            try:
                column = self.rowdict[row]
                return range(min(column), max(column) + 1)
            except KeyError:
                return range(0)

    def __init__(self):
        self.shape_collection = []
        if MOD_SETTINGS['DEBUG']:
            self.registration_tracker = []

    def _straight_cross(self, edge1, edge2):
        """Find all crossings of two given straight lines."""
        nodes = []
        crits = []
        # There are multiple special cases to be considered here
        # 1: Are the base objects of both edges geometrically
        # identical?
        if (edge1.source.get_shape_data()[1:6]
                == edge2.source.get_shape_data()[1:6]):
            # Obviously, identical lines do not really have a distinct
            # crossing.  For the algorithm to not generate useless or
            # random results, we exclusively consider the start and end
            # of the older edge as possible crossing points between the
            # two.  An additional requirement is that it is not at the
            # same time the startpoint or endpoint of the new edge.
            # Note that it will get attached to the list of critical
            # crossings (the ones near corners) in order to properly
            # consider and discard crossing information coming from the
            # adjacent edge.
            sp_cross = edge2.get_point_on_straight(edge1.startpoint.coords)
            ep_cross = edge2.get_point_on_straight(edge1.endpoint.coords)
            if sp_cross is not None:
                sp_node = Node(edge1.startpoint.coords)
                crits.append((edge1, 0.0, edge2, sp_cross, sp_node, 0.0))
            if ep_cross is not None:
                ep_node = Node(edge1.endpoint.coords)
                crits.append((edge1, 1.0, edge2, ep_cross, ep_node, 0.0))
        # 2: Are the edges parallel?
        elif abs(edge1.direction[0] / edge2.direction[0]
                 - edge1.direction[1] / edge2.direction[1]) < 1e-8:
            pass
        # 3: If none of the above was the case, the two edges have to
        # cross somewhere.  So we calculate where exactly they do.
        else:
            # This try...except LinAlgError should be obsolete due to
            # the check above, but for bugfixing purposes it shall
            # remain for now.
            try:
                params = _straight_cross(sp1=edge1.startpoint.coords,
                                         sp2=edge2.startpoint.coords,
                                         dir1=edge1.direction,
                                         dir2=edge2.direction)
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix slipped through despite "
                      "checking")
            else:
                legal1, crit1 = edge1.check_within_bounds(params[0])
                legal2, crit2 = edge2.check_within_bounds(params[1])
                if legal1 and legal2:
                    split_point = edge1.calc_split_point(params[0])
                    node = Node(split_point)
                    if not (crit1 or crit2):
                        edge1.splits.append((params[0], node))
                        edge2.splits.append((params[1], node))
                        nodes.append(node)
                    else:
                        grad1 = edge1.get_gradient()
                        grad2 = edge2.get_gradient()
                        scp = _scalar_product(grad1, grad2)
                        crits.append((edge1, params[0], edge2, params[1], node, scp))
        return nodes, crits

    def _straight_arc_cross(self, straight, arc):
        """Find all crossings between a straight line and an arc."""
        nodes = []
        crits = []
        # support and directional vectors of the straight line,
        # transformed into the bound coordinate system of the ellipse.
        sup = arc.transform(arc.translate(straight.startpoint.coords, True),
                            True)
        drc = arc.transform(straight.direction, True)
        ellipse = arc.source
        # Factors for the midnight formula
        a = drc[0] ** 2 * ellipse.len_2 ** 2 + drc[1] ** 2 * ellipse.len_1 ** 2
        b = 2 * (sup[0] * drc[0] * ellipse.len_2 ** 2 + sup[1] * drc[1] * ellipse.len_1 ** 2)
        c = (sup[0] ** 2 * ellipse.len_2 ** 2 + sup[1] ** 2 * ellipse.len_1 ** 2
             - ellipse.len_1 ** 2 * ellipse.len_2 ** 2)
        # Calculate the roots via the midnight formula
        radix = b ** 2 - 4.0 * a * c
        if radix <= 0.0:
            # We can stop early when no real solutions exist.
            return nodes, crits
        sols = ((-b + math.sqrt(radix)) / (2.0 * a),
                (-b - math.sqrt(radix)) / (2.0 * a))
        for sol in sols:
            valid1, crit1 = straight.check_within_bounds(sol)
            if valid1:
                split_point = straight.calc_split_point(sol)
                valid2, crit2, param = arc.check_within_bounds(split_point)
                if valid2:
                    node = Node(split_point)
                    if not (crit1 or crit2):
                        straight.splits.append((sol, node))
                        arc.splits.append((param, node))
                        nodes.append(node)
                    else:
                        straight_grad = straight.get_gradient(split_point)
                        arc_grad = arc.get_gradient(split_point)
                        scp = _scalar_product(straight_grad, arc_grad)
                        crits.append((straight, sol, arc, param, node, scp))
        return nodes, crits

    def _arc_cross(self, arc1, arc2):
        """Find all crossings between two arcs.
        The following code roughly implements the algorithm described
        by David Eberly in his paper "Intersection of Ellipses",
        available under
        [1]: https://www.geometrictools.com/Documentation/IntersectionOfEllipses.pdf
        """

        def _y_transform(w, x):
            return w - (poly1[2] + x * poly1[4]) / 2.0

        # Since the following algorithms tend to give somewhat inexact
        # results due to numeric uncertainties and rounding errors, we
        # introduce a safety margin of what we realistically consider to
        # be zero.
        zero_threshold = 1e-10
        nodes = []
        crits = []
        # Fetch parameters of the associated ellipses
        poly1 = arc1.source.fetch_implicit()
        poly2 = arc2.source.fetch_implicit()
        # Subtract the two polynomial vectors to get an equation
        # representing the difference:
        # (diff[0] + diff[1]*x + diff[2]*y
        #  + diff[3]*x**2 + diff[4]*x*y) = 0
        diff = 5 * [None]
        for index in range(len(poly1)):
            c_diff = poly1[index] - poly2[index]
            # Very small differences should be safely ignorable as
            # testing showed that they tend to be numerical artifacts
            # that can furthermore throw the following algorithm
            # severely off track.
            if abs(c_diff) < zero_threshold:
                c_diff = 0.0
            diff[index] = c_diff
        # If the difference is zero in all entries, the ellipses are
        # identical and we can stop the regular algorithm early.
        # Instead, we specifically only consider the start and end
        # point of the older arc (always arc1) as crosspoints, and only
        # if they have been changed from the standard -pi/pi.
        # Background: If they are in fact pi or -pi, this points to
        # them being unchanged, but we do not want to consider
        # crosspoints where there is no border to another edge.
        if max(diff) == 0.0 and min(diff) == 0.0:
            if arc1.begin not in {math.pi, -math.pi}:
                node = Node(arc1.startpoint.coords)
                _, _, param = arc2.check_within_bounds(arc1.startpoint.coords)
                crits.append((arc1, 0.0, arc2, param, node, 0.0))
            if arc1.end not in {math.pi, -math.pi}:
                node = Node(arc1.endpoint.coords)
                _, _, param = arc2.check_within_bounds(arc1.endpoint.coords)
                crits.append((arc1, 1.0, arc2, param, node, 0.0))
            return nodes, crits
        # Using the transformation proposed in [1], the polynomials
        # necessary for the actual calculations are called c, d and e,
        # the parameters of which are calculated as follows.
        c = poly.Polynomial([poly1[0] - ((poly1[2] ** 2) / 4.0),
                             poly1[1] - (poly1[2] * poly1[4] / 2.0),
                             poly1[3] - ((poly1[4] ** 2) / 4.0)])
        d = poly.Polynomial([diff[2], diff[4]])
        e = poly.Polynomial([diff[0] - (poly1[2] * diff[2] / 2.0),
                             diff[1] - (((poly1[2] * diff[4])
                                         + (poly1[4] * diff[2])) / 2.0),
                             diff[3] - (poly1[4] * diff[4] / 2.0)])
        try:
            xbar = -diff[2] / diff[4]
        except ZeroDivisionError:
            xbar = 0.0
        points = []
        # For the actual calculation, we need to switch cases.
        # Out of the cases found in [1], the following branch deals
        # with case (1)
        if abs(e(xbar)) > zero_threshold and diff[4] != 0.0:
            f = (c * (d ** 2)) + (e ** 2)
            roots = poly.polyroots(f.coef)
            # TEST: Further optimize the results by performing a newton
            # optimization on the results.
            fprime = f.deriv()
            fprime2 = fprime.deriv()
            for i in range(len(roots)):
                roots[i] = opti.newton(f, roots[i], fprime, tol=1e-8,
                                       fprime2=fprime2, disp=False)
            # As described in the documentation of polyroots, it
            # returns 'float64' type arrays if all roots are real and
            # 'complex128' type otherwise.  However, we also need to
            # deal with cases where only some roots are complex, so we
            # need to consider each root on its own.  To do so, we cast
            # to complex if not already the case and discard values
            # depending on the imaginary part in the for-loop below.
            if roots.dtype == np.dtype('f8'):
                roots = roots.astype('c16')
            for root in roots:
                # Complex roots are irrelevant to us, so we discard
                # them. Due to numerical concerns with the root finding
                # of numpy.polynomial, we give a threshold for the
                # imaginary part being "virtually zero".
                if abs(root.imag) < zero_threshold:
                    w = -e(root.real) / d(root.real)
                    y = _y_transform(w, root.real)
                    points.append((root.real, y))
        # Out of the cases found in [1], the following branch deals
        # with case (2)
        elif abs(e(xbar)) < zero_threshold and diff[4] != 0.0:
            if c(xbar) < -zero_threshold:
                w = math.sqrt(-c(xbar))
                y = _y_transform(w, xbar)
                points.append((xbar, y))
                w = -w
                y = _y_transform(w, xbar)
                points.append((xbar, y))
            h = poly.Polynomial([((e.coef[1] - (diff[2] * e.coef[2] / diff[4]))
                                  / diff[4]),
                                 e.coef[2] / diff[4]])
            f = h ** 2 + c
            roots = poly.polyroots(f.coef)
            # TEST: Further optimize the results by performing a newton
            # optimization on the results.
            fprime = f.deriv()
            fprime2 = fprime.deriv()
            for i in range(len(roots)):
                roots[i] = opti.newton(f, roots[i], fprime, tol=1e-8,
                                       fprime2=fprime2, disp=False)
            if roots.dtype == np.dtype('f8'):
                for root in roots:
                    w = -h(root)
                    y = _y_transform(w, root)
                    points.append((root, y))
        # Out of the cases found in [1], the following branch deals
        # with cases (3) and (4)
        elif diff[4] == 0.0 and diff[2] != 0.0:
            f = (c * (d ** 2)) + (e ** 2)
            roots = poly.polyroots(f.coef)
            # As described in the documentation for polyroots, it
            # returns 'float64' type arrays if all roots are real and
            # 'complex128' type otherwise.  However, we also need to
            # deal with cases where only some roots are complex, so we
            # need to consider each root on its own.  To do so, we cast
            # to complex if not already the case and discard values
            # depending on the imaginary part in the for-loop below.
            if roots.dtype == np.dtype('f8'):
                roots = roots.astype('c16')
            # TEST: Further optimize the results by performing a newton
            # optimization on the results.
            fprime = f.deriv()
            fprime2 = fprime.deriv()
            for i in range(len(roots)):
                roots[i] = opti.newton(f, roots[i], fprime, tol=1e-8,
                                       fprime2=fprime2, disp=False)
            for root in roots:
                # Complex roots are irrelevant to us, so we discard
                # them. Due to numerical concerns with the root finding
                # of numpy.polynomial, we give a threshold for the
                # imaginary part being "practically zero".
                if abs(root.imag) < zero_threshold:
                    w = -e(root.real) / diff[2]
                    y = _y_transform(w, root.real)
                    points.append((root.real, y))
        elif diff[4] == 0.0 and diff[2] == 0.0:
            if abs(e.coef[2]) > zero_threshold:
                f = [e.coef[0] / e.coef[2], e.coef[1] / e.coef[2]]
                delta = f[1] ** 2 / 4.0 - f[0]
                if delta > 0.0:
                    xhat = -f[1] / 2.0 + math.sqrt(delta)
                    if c(xhat) < 0.0:
                        w = math.sqrt(-c(xhat))
                        y = _y_transform(w, xhat)
                        points.append((xhat, y))
                        w = -w
                        y = _y_transform(w, xhat)
                        points.append((xhat, y))
                    xhat = -f[1] / 2.0 - math.sqrt(delta)
                    if c(xhat) < 0.0:
                        w = math.sqrt(-c(xhat))
                        y = _y_transform(w, xhat)
                        points.append((xhat, y))
                        w = -w
                        y = _y_transform(w, xhat)
                        points.append((xhat, y))
            else:
                if abs(e.coef[1]) < zero_threshold:
                    return []
                xhat = -e.coef[0] / e.coef[1]
                if c(xhat) < 0.0:
                    w = math.sqrt(-c(xhat))
                    y = _y_transform(w, xhat)
                    points.append((xhat, y))
                    w = -w
                    y = _y_transform(w, xhat)
                    points.append((xhat, y))
        else:
            raise NotImplementedError("More cases need to be considered "
                                      "calculating the ellipse-ellipse-"
                                      "intersection.")
        # Since there are no guarantees concerning the numerical
        # stability of the algorithm up until this point, we do another
        # round of checking to discard points that turned out to be
        # very close together.  These do not contribute much even if
        # they constitute real crossings and can actively harm the
        # operation if they are just numerical artifacts.
        index = 0
        while index < len(points):
            for compare in range(index + 1, len(points)):
                if (abs(points[index][0] - points[compare][0])
                    + abs(points[index][1] - points[compare][1])) < 1e-3:
                    del points[compare]
                    del points[index]
                    index = -1
                    break
            index += 1
        for point in points:
            valid1, crit1, param1 = arc1.check_within_bounds(point)
            valid2, crit2, param2 = arc2.check_within_bounds(point)
            if valid1 and valid2:
                node = Node(point)
                if not (crit1 or crit2):
                    arc1.splits.append((param1, node))
                    arc2.splits.append((param2, node))
                    nodes.append(node)
                else:
                    grad1 = arc1.get_gradient(point)
                    grad2 = arc2.get_gradient(point)
                    scp = _scalar_product(grad1, grad2)
                    crits.append((arc1, param1, arc2, param2, node, scp))
        return nodes, crits

    def _get_cross_function(self, oldedge, newedge):
        """Dispatches the two edges to the respective intersection
        finders.
        """
        if isinstance(oldedge, Straight) and isinstance(newedge, Straight):
            nodes, crits = self._straight_cross(oldedge, newedge)
        elif isinstance(oldedge, Arc) and isinstance(newedge, Arc):
            nodes, crits = self._arc_cross(oldedge, newedge)
        # Note for _straight_arc_cross: It assumes that its first
        # argument is the straight line and its second argument is the
        # elliptical arc.  That is why there are two separate
        # dispatches for it, once swapping oldedge and newedge and once
        # not.
        elif isinstance(oldedge, Straight) and isinstance(newedge, Arc):
            nodes, crits = self._straight_arc_cross(oldedge, newedge)
        elif isinstance(oldedge, Arc) and isinstance(newedge, Straight):
            nodes, crits = self._straight_arc_cross(newedge, oldedge)
        else:
            raise NNLibUsageError("Unrecognized edge type.")
        return nodes, crits

    def _process_critical_points(self, crits):
        """ Note: Separated from 'register_shape' to keep it simple.
        This function deals with critical crossing points that were
        found while scanning for edge intersections.  Returns a list of
        nodes that are to be added to the list of nodes for further
        processing.
        """
        nodes = []
        while crits:
            sim_coll = [crits.pop()]
            i = 0
            # Collect all crossings that are very close to the first.
            while i < len(crits):
                if (abs(crits[i][4].coords[0]
                        - sim_coll[0][4].coords[0])
                        + abs(crits[i][4].coords[1]
                              - sim_coll[0][4].coords[1]) < 1e-8):
                    sim_coll.append(crits.pop(i))
                else:
                    i += 1
            # There are multiple empirical rules for deciding what to
            # do with these crossings.
            if len(sim_coll) == 1:
                # In the case of just one crossing having been
                # marked as critical, we accept it into the list of
                # crossings if and only if the associated
                # parameters are valid. (The assumption here is
                # that they are either normal crossings or close
                # misses that just happened near corners by
                # chance.)
                cross = sim_coll[0]
                if 0.0 < cross[1] <= 1.0 and 0.0 <= cross[3] <= 1.0:
                    nodes.append(cross[4])
                    cross[0].splits.append((cross[1], cross[4]))
                    cross[2].splits.append((cross[3], cross[4]))
            elif len(sim_coll) == 2:
                # In the case of two crossings that are very close
                # together, we discard cases where one points into
                # and another points out of the crossing shape.
                # This leads to us potentially ignoring some tiny
                # areas, but we believe this should not matter in
                # the end.
                cross1, cross2 = sim_coll
                if cross1[5] * cross2[5] >= 0.0:
                    if (0.0 <= cross1[1] <= 1.0
                            and 0.0 <= cross1[3] <= 1.0):
                        cross = cross1
                    elif (0.0 <= cross2[1] <= 1.0
                          and 0.0 <= cross2[3] <= 1.0):
                        cross = cross2
                    else:
                        c1p1 = max(-cross1[1], cross1[1] - 1.0, 0.0)
                        c1p2 = max(-cross1[3], cross1[3] - 1.0, 0.0)
                        c2p1 = max(-cross2[1], cross2[1] - 1.0, 0.0)
                        c2p2 = max(-cross2[3], cross2[3] - 1.0, 0.0)
                        if c1p1 + c1p2 < c2p1 + c2p2:
                            cross = cross1
                        else:
                            cross = cross2
                    nodes.append(cross[4])
                    cross[0].splits.append((cross[1], cross[4]))
                    cross[2].splits.append((cross[3], cross[4]))
            else:
                # The case of more than two crossings that are very
                # close together is quite rare and, so far, no
                # satisfyingly accurate rule was found, so, for now, we
                # just throw an error.
                raise NotImplementedError("Cases for more than two similar "
                                          "points have not yet been "
                                          "formulated.")
        # End of while loop
        return nodes

    def register_shape(self, new_shape):
        """CHANGED in mo_renderer_new_v09: nnlib now supports different
        intensities per object. To reflect this, this method is changed
        such that object are no longer fused, but still intersected.
        Registers new shapes.  Automatically discards invisible and
        ineffectual objects.  Reminder: The order in which shapes are
        added greatly influences the resulting field as the fusing is
        implicitly hierarcial by nature.  Think of it as if the objects
        get located increasingly closer to your eye, potentially
        obscuring what is behind them.
        """
        if MOD_SETTINGS['DEBUG']:
            # For debugging purposes, we save infos about incoming
            # objects for future reference before any processing.
            self.registration_tracker.append(new_shape.get_shape_data())
        # Step 1: Find all the crossings and split the objects at the
        # found nodes (if any).
        nodes = []
        for obj in self.shape_collection:
            crits = []
            for edge1 in obj.edge_list:
                for edge2 in new_shape.edge_list:
                    # Due to continuing numerical problems in
                    # situations where a crossing happens near an
                    # object corner, we will from now on separate said
                    # critical instances and consider them further when
                    # all crossings have been defined.
                    nnodes, ncrits = self._get_cross_function(edge1, edge2)
                    nodes += nnodes
                    crits += ncrits
            # Check and discard or accept critical nodes
            if crits:
                nodes += self._process_critical_points(crits)
            obj.split()
        new_shape.split()
        # Step 2: Find all viable vector lines that run around closed
        # shapes in the multiobject.  To do this, we go through all
        # previously generated nodes.  We stop immediately when we
        # encounter a known line to save us from excessive amounts of
        # vectorlines.
        new_vectorlines = []
        if new_shape.fused:
            for node in nodes:
                for start_index in range(1):
                    if not node.check(start_index):
                        vl = VectorLine()
                        if vl.trace(node, start_index):
                            new_vectorlines.append(vl)
        else:
            new_vectorlines.append(new_shape)
        # -- New algorithm for sorting out shapes begins here --
        # Step 3a: Add all old vectorlines that neither were fused nor
        # live inside the new shape to the list of vectorlines to
        # consider further.
        for shape in self.shape_collection:
            if (not shape.fused
                    and (shape.area > new_shape.area
                         or not new_shape.check_inside(shape.centerpoint))):
                new_vectorlines.append(shape)
        # Step 3b: Filter down the vectorlines.  To achieve that:
        # Sort after area (starting with the smallest)
        new_vectorlines.sort(key=lambda x: x.area)
        index = 0
        while index < len(new_vectorlines):
            current = new_vectorlines[index]
            for comp in new_vectorlines[index + 1:]:
                # We ensured current to be smaller than comp, so we
                # check whether comp contains current.
                if comp.check_inside(current.centerpoint):
                    # If so, comp entirely contains current.  If they
                    # have the same additivity, current can be safely
                    # discarded without losing info.
                    if (comp.additive == current.additive):
                        del new_vectorlines[index]
                    # If they have different additivity, current lives
                    # inside comp as a white (or black, depending on
                    # combination) spot and needs to stay.
                    else:
                        index += 1
                    # In both cases, however, we can stop considering
                    # current.
                    break
                # The loop continues if comp does not contain current.
            else:
                # The else clause is executed if the for-loop has not
                # been broken, which means current is not contained
                # within any other shape.
                # In this case, we delete if current is not additive,
                # and leave it if it is.
                if current.additive:
                    index += 1
                else:
                    del new_vectorlines[index]
        # Step 3c: Write the filtered out list of shapes to the scope's
        # shape collection.
        self.shape_collection = new_vectorlines
        return

    def draw(self, array_access):
        debug_strings = ''
        field_proto = self.FieldProto()
        for shape in self.shape_collection:
            shape.draw(field_proto)
        field_proto.close()
        for row in field_proto.row_iter():
            fill = False
            for col in field_proto.col_iter(row):
                if field_proto[row, col]:
                    pixel = (row, col)
                    rlist, in_area = make_relative_list(field_proto[row, col],
                                                        pixel)
                    if not rlist and fill:
                        # This fixes a rare bug occurrence subtractive
                        # areas exclusively inside one pixel and no
                        # other borders on the pixel could yield
                        # negative results.  Now we take into account
                        # whether the surrounding pixels are otherwise
                        # supposed to be filled.  If yes, it is
                        # appropriate to set the base area coverage to
                        # 1.0 (2.0 because it will be halved again
                        # later).
                        area = 2.0
                    else:
                        area, fill = integrate_border(rlist, pixel)
                    total_area = 0.5 * (area + in_area)
                    if total_area < 0.0 or total_area > 1.0:
                        debug_strings += ("Value of %f encountered at x_pos "
                                          "%i, y_pos %i."
                                          % (total_area, col, row))
                    array_access(total_area, row, col)
                elif fill:
                    array_access(1.0, row, col)
        if debug_strings:
            warnings.warn(debug_strings)
        else:
            return


shape_map = {'ellipse': Ellipse,
             'rectangle': Rectangle}


class LegacyInterface(RenderInterface):
    """This class interfaces with nnlib's own original rendering
    solution, the code of which lives in mo_renderer_new.
    """

    def check(array_access, thread_num, field_size, field_depth):
        if field_depth == 1:
            return LegacyInterface(array_access, field_size)
        else:
            return None

    def __init__(self, array_access, field_size):
        """LegacyInterface doesn't need to know thread_num, field_size
        or field_depth, but BlenderInterface does, so we take it and
        drop it here if supplied.
        """
        super(LegacyInterface, self).__init__(array_access)
        self.scope = FieldScope()
        self.field_size = field_size

    def submit_obj(self, coords, shape, presence):
        # CHANGED with mrn_11 as a result from mo_08 changes: It is the
        # interface's responsibility to scale the normalized
        # coordinates, which we do here.
        coords[:ang_index] *= self.field_size
        coords[ang_index:] *= math.pi / 2.0
        shape_obj = shape_map[shape](*coords, additive=presence)
        self.scope.register_shape(shape_obj)

    def finalize(self):
        self.scope.draw(self.dtarget)
        # To make the interaction as simple as possible, the scopes get
        # renewed implicitly from the caller's point of view.  This
        # allows other render interfaces to not renew their scope if
        # that doesn't make sense for them.
        self.scope = FieldScope()


register_render_if(LegacyInterface)

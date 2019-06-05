"""

MB-DPOP algorithm

An extended for the DPOP algorithm which is based on the paper
Petcu and Faltings 2006 https://infoscience.epfl.ch/record/98347


This paper introduce a a new hybrid algorithm that is controlled by a parameter k which
characterizes the amount of available memory. K specifies the maximal amount of inference
or the maximal message dimensionality. The parameter is chosen such that the variable
memory at each node is higher than d**k. (d is domain size)

If k >= w (w is the induced width), full inference is done throughout the problem.
If k = 1, only linear message are used.
If k < w, full inference is done in the areas of width lower than k, and bounded inference
in areas of width higher than k.

--------------


"""

from random import choice
from typing import Iterable
import itertools

from pydcop.infrastructure.computations import Message, VariableComputation, register
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import NAryMatrixRelation, RelationProtocol, \
    Constraint, get_data_type_max, get_data_type_min, generate_assignment, \
    generate_assignment_as_dict, filter_assignment_dict, find_arg_optimal, \
    DEFAULT_TYPE
from pydcop.algorithms import AlgoParameterDef, ALGO_STOP, ALGO_CONTINUE, ComputationDef, \
    check_param_value

GRAPH_TYPE = 'pseudotree'

"""
MB DPOP supports one paramater: 
* avail_mem

default to be a large int, then MB-DPOP is normal dpop.
"""
algo_params = [
    AlgoParameterDef("avail_mem", "int", 2 ** 32 - 1),
]

k = 2


def build_computation(comp_def: ComputationDef):
    """
    takes in a computation definition and build the dpop algorithm by pass them to DpopAlgo

    :param comp_def:
    :return: computation = DpopAlgo(...)
    """

    parent = None
    children = []
    pseudo_parent = []
    pseudo_children = []
    for l in comp_def.node.links:
        if l.type == 'parent' and l.source == comp_def.node.name:
            parent = l.target
        if l.type == 'children' and l.source == comp_def.node.name:
            children.append(l.target)
        """
        The pseudo relations are added for MB-DPOP
        """
        if l.type == 'pseudo_parent' and l.source == comp_def.node.name:
            pseudo_parent.append(l.target)
        if l.type == 'pseudo_children' and l.source == comp_def.node.name:
            pseudo_children.append(l.target)

    constraints = [r for r in comp_def.node.constraints]

    computation = MBDpopAlgo(comp_def.node.variable, parent,
                           children, pseudo_parent, pseudo_children, constraints,
                           comp_def=comp_def)
    return computation


class MBDpopMessage(Message):
    """
    A class for MB-DPOP message

    :
    TODO

    It is unclear if an extension is needed


    """

    def __init__(self, msg_type, content):
        super(MBDpopMessage, self).__init__(msg_type, content)

    @property
    def size(self):
        # MBDpop messages
        # UTIL : context & multi-dimensional matrices
        # VALUE : a value assignment for each var in the
        #             separator of the sender
        # LABEL : 2 lists, for separator and cycle_cuts

        if self.type == 'UTIL':
            # UTIL messages are multi-dimensional matrices
            shape = self.content[1].shape
            size = 1
            for s in shape:
                size *= s
            return size

        elif self.type == 'VALUE':
            # VALUE message are a value assignment for each var in the
            # separator of the sender
            return len(self.content[0]) * 2

        elif self.type == 'LABEL':
            # LABEL message are 2 lists, for separator and cycle_cuts
            return len(self.content[0]) + len(self.content[1])

    def __str__(self):
        return 'MBDpopMessage({}, {})'.format(self._msg_type, self._content)


def join_utils(u1: Constraint, u2: Constraint) -> Constraint:
    """
    Build a new relation by joining the two relations u1 and u2.

    The dimension of the new relation is the union of the dimensions of u1
    and u2. As order is important for most operation, variables for u1 are
    listed first, followed by variables from u2 that where already used by u1
    (in the order in which they appear in u2.dimension).

    For any complete assignment, the value of this new relation is the sum of
    the values from u1 and u2 for the subset of this assignment that apply to
    their respective dimension.

    For more details, see the definition of the join operator in Petcu Phd
    Thesis.

    :param u1: n-ary relation
    :param u2: n-ary relation
    :return: a new relation
    """
    #
    dims = u1.dimensions[:]
    for d2 in u2.dimensions:
        if d2 not in dims:
            dims.append(d2)

    u_j = NAryMatrixRelation(dims, name='joined_utils')
    for ass in generate_assignment_as_dict(dims):
        # FIXME use dict for assignement
        # for Get AND sett value

        u1_ass = filter_assignment_dict(ass, u1.dimensions)
        u2_ass = filter_assignment_dict(ass, u2.dimensions)
        s = u1(**u1_ass) + u2(**u2_ass)
        u_j = u_j.set_value_for_assignment(ass, s)

    return u_j


def projection(a_rel, a_var, mode='max', a_var_val=""):
    """

    The project of a relation a_rel along the variable a_var is the
    optimization of the matrix along the axis of this variable.

    The result of `projection(a_rel, a_var)` is also a relation, with one less
    dimension than a_rel (the a_var dimension).
    each possible instantiation of the variable other than a_var,
    the optimal instantiation for a_var is chosen and the corresponding
    utility recorded in projection(a_rel, a_var)

    Also see definition in Petcu 2007

    :param a_rel: the projected relation
    :param a_var: the variable over which to project
    :param mode: 'max (default) for maximization, 'min' for minimization,
                    'context' for passing an assignment
    :param a_var_val: the context value for variable a_var if context mode is set.

    :return: the new relation resulting from the projection
    """

    remaining_vars = a_rel.dimensions.copy()
    remaining_vars.remove(a_var)

    # the new relation resulting from the projection
    proj_rel = NAryMatrixRelation(remaining_vars)

    all_assignments = generate_assignment(remaining_vars)
    for partial_assignment in all_assignments:
        # for each assignment, look for the max value when iterating over
        # aVar domain

        if mode != "context":
            if mode == 'min':
                best_val = get_data_type_max(DEFAULT_TYPE)
            else:
                best_val = get_data_type_min(DEFAULT_TYPE)

            for val in a_var.domain:
                full_assignment = _add_var_to_assignment(partial_assignment,
                                                         a_rel.dimensions, a_var,
                                                         val)

                current_val = a_rel.get_value_for_assignment(full_assignment)
                if (mode == 'max' and best_val < current_val) or \
                        (mode == 'min' and best_val > current_val):
                    best_val = current_val
        else:
            if a_var_val == "":
                raise ValueError("Model context has been set but value is not passed in.")
            best_val = a_var_val

        proj_rel = proj_rel.set_value_for_assignment(partial_assignment,
                                                     best_val)

    return proj_rel


def _add_var_to_assignment(partial_assignt, ass_vars, new_var, new_value):
    """
    Add a value for a variable in an assignment.
    The given partial assignment is not modified and a new assignment is
    returned, augmented with the value for the new variable, in the right
    position according to `ass_vars`.

    :param partial_assignt: a partial assignment represented as a list of
    values, the order of the values maps the order of the corresponding
    variables in `ass_vars`
    :param ass_vars: a list of variables corresponding to the list to the
    variables whose values are given by `partial_assignt`, augmented with one
    extra variable 'new_var' whose value is given by `new_value`.
    :param new_var: variable that must be added in the assignment
    :param new_value: value to add in the assignement for the new variable

    """

    if len(partial_assignt) + 1 != len(ass_vars):
        raise ValueError('Length of partial assignment and variables do not '
                         'match.')
    full_assignment = partial_assignt[:]
    for i in range(len(ass_vars)):
        if ass_vars[i] == new_var:
            full_assignment.insert(i, new_value)
    return full_assignment


class MBDpopAlgo(VariableComputation):
    """
    Dynamic programming Optimization Protocol

    This class represents the MB-DPOP algorithm.

    When running this algorithm, the DFS tree must be already defined and the
    children, parents and pseudo-parents must be known.

    Three kind of messages:
    * UTIL message:
      sent from children to parent, contains a relation (as a
      multi-dimensional matrix) with one dimension for each variable in our
      separator.
    * VALUE messages :
      contains the value of the parent of the node and the values of all
      variables that were present in our UTIl message to our parent (that is
      to say, our separator) .
    * LABEL messages :
      contains 2 lists, for separator and cycle_cuts
    """

    def __init__(self,
                 variable: Variable,
                 parent: str,
                 children: Iterable[str],
                 pseudo_parent: Iterable[str],
                 pseudo_children: Iterable[str],
                 constraints: Iterable[RelationProtocol],
                 msg_sender=None, comp_def=None):
        """

        In DPOP,
        * a relation is managed by a single agent (i.e. algorithm object in
        our case)
        * a relation must always be managed by the lowest node in the DFS
        tree that the relation depends on (which is especially important for
        non-binary relation).


        :param variable: The Variable object managed by this algorithm

        :param parent: the parent for this node. A node has at most one parent
        but may have 0-n pseudo-parents. Pseudo parent are not given
        explicitly but can be deduced from the relation set with add_relation.
        If the variable shares a constraints with its parent (which is the
        most common case), it must be present in the relation arg.

        :param children: the children variables of the variable arguemnt,
        in the DFS tree

        :param constraints: relations managed by this computation. These
        relation will be used when calculating costs. It must
        depends on the variable arg. Unary relation are also supported.
        Remember that a relation must always be managed by the lowest node in
        the DFS tree that the relation depends on (which is especially
        important for non-binary relation).

        :param msg_sender: the object that will be used to send messages to
        neighbors, it must have a  post_msg(sender, target_name, name) method.

        :param mode: type of optimization to perform, 'min' or 'max'
        """
        super().__init__(variable, comp_def)

        assert comp_def.algo.algo == 'dpop'

        self._msg_sender = msg_sender
        self._mode = comp_def.algo.mode
        self._parent = parent
        self._children = children
        self._constraints = constraints

        if hasattr(self._variable, 'cost_for_val'):
            costs = []
            for d in self._variable.domain:
                costs.append(self._variable.cost_for_val(d))
            self._joined_utils = NAryMatrixRelation([self._variable], costs,
                                                    name='joined_utils')

        else:
            self._joined_utils = NAryMatrixRelation([], name='joined_utils')

        self._children_separator = {}

        self._waited_children_util = []
        if not self.is_leaf:
            # If we are not a leaf, we must wait for the util messages from
            # our children.
            # This must be done in __init__ and not in on_start because we
            # may get an util message from one of our children before
            # running on_start, if this child computation start faster of
            # before us
            self._waited_children_util = self._children[:]

        """
        The following fields are added for MB-DPOP
        Initialize the separators as the union of parent and pseudoparent, 
        separators might have different definitions for the parts from original DPOP 
        implementation. 
        """
        self._pparent = pseudo_parent
        self._pchildren = pseudo_children
        # sep_i in the paper
        self._separators = [x for x in set(pseudo_parent) | set(self._parent)]
        # cc_lists in the paper, hasn't been unioned
        self._cc_children = []
        # cc_i in the paper
        self._cc = []
        self._waited_children_label = self._waited_children_util
        self._label_CR= "undecided"
        self._cache = {}

    def footprint(self):
        return computation_memory(self.computation_def.node)

    @property
    def is_root(self):
        return self._parent is None

    @property
    def is_leaf(self):
        return len(self._children) == 0

    @property
    def is_stable(self):
        return False

    def on_start(self):
        msg_count, msg_size = 0, 0

        if self.is_leaf and not self.is_root:
            # If we are a leaf in the DFS Tree we can immediately compute
            # our util and send it to our parent.
            # Note: as a leaf, our separator is the union of our parents and
            # pseudo-parents
            util = self._compute_utils_msg()
            self.logger.info('Leaf {} init message {} -> {}  : {}'.format(
                             self._variable.name, self._variable.name,
                             self._parent, util))
            msg = MBDpopMessage('UTIL', [[] ,util])
            self.post_msg(self._parent, msg)
            msg_count += 1
            msg_size += msg.size

        elif self.is_leaf:
            # we are both root and leaf : means we are a isolated variable we
            #  can select our own value alone:
            if self._constraints:
                for r in self._constraints:
                    self._joined_utils = join_utils(self._joined_utils, r)

                values, current_cost = find_arg_optimal(
                    self._variable, self._joined_utils, self._mode)

                self.select_value_and_finish(values[0], float(current_cost))
            else:
                # If the variable is not constrained, we can simply take a value at
                # random:
                value = choice(self._variable.domain)
                self.select_value_and_finish(value, 0.0)

    def stop_condition(self):
        # dpop stop condition is easy at it only selects one single value !

        #TODO
        # If the Condition also holds for MB_DPOP
        if self.current_value is not None:
            return ALGO_STOP
        else:
            return ALGO_CONTINUE

    def select_value_and_finish(self, value, cost):
        """
        Select a value for this variable.

        DPOP is not iterative, once we have selected our value the algorithm
        is finished for this computation.

        Parameters
        ----------
        value: any (depends on the domain)
            the selected value
        cost: float
            the local cost for this value

        """

        self.value_selection(value, cost)
        self.stop()
        self.finished()
        self.logger.info('Value selected at %s : %s - %s', self.name,
                         value, cost)

    @register("UTIL")
    def _on_util_message(self, variable_name, recv_msg, t):
        self.logger.debug(f'Util message from {variable_name} : {recv_msg.content} ')
        context, utils = recv_msg.content
        msg_count, msg_size = 0, 0

        """
        This part is done according to the Algo 1 UTIL propagation protocol
        """
        if self._label_CR == "normal":
            # it is a normal node

            # accumulate util messages until we got the UTIL from all our children
            self._joined_utils = join_utils(self._joined_utils, utils)
            try:
                self._waited_children_util.remove(variable_name)
            except ValueError as e:
                self.logger.error(f'Unexpected UTIL message from {variable_name} on {self.name} : {recv_msg} ')
                raise e
            # keep a reference of the separator of this children, we need it when
            # computing the value message
            self._children_separator[variable_name] = utils.dimensions

            if len(self._waited_children_util) == 0:

                if self.is_root:
                    # We are the root of the DFS tree and have received all utils
                    # we can select our own value and start the VALUE phase.

                    # The root obviously has no parent nor pseudo parent, yet it
                    # may have unary relations (with it-self!)
                    for r in self._constraints:
                        self._joined_utils = join_utils(self._joined_utils, r)

                    values, current_cost = find_arg_optimal(
                        self._variable, self._joined_utils, self._mode)
                    selected_value = values[0]

                    self.logger.info(f'ROOT: On UNTIL message from {variable_name}, send value \
                                        msg to childrens {self._children} ')
                    for c in self._children:
                        msg = MBDpopMessage('VALUE', ([self._variable],
                                                      [selected_value]))
                        self.post_msg(c, msg)
                        msg_count += 1
                        msg_size += msg.size

                    self.select_value_and_finish(selected_value,
                                                 float(current_cost))
                else:
                    # We have received the Utils msg from all our children, we can
                    # now compute our own utils relation by joining the accumulated
                    # util with the relations with our parent and pseudo_parents.
                    util = self._compute_utils_msg()
                    msg = MBDpopMessage('UTIL', [[], util])
                    self.logger.info(f'On UTIL message from {variable_name}, send UTILS msg '
                                     f'to parent  { self._children}')
                    self.post_msg(self._parent, msg)
                    msg_count += 1
                    msg_size += msg.size
        else:
            # the node is abnormal node

            # do propagations for all instantiation of CClist
            for instantiation in self._cache.keys():
                # a value assignment
                context = { self._separators[i]:instantiation[i] for i in range(instantiation)}

                # do propagation

            if self._label_CR == "CR":

                # update UTIL and Cache for each propagation

                # when propagation finishes , send UTIL to parents


    def _compute_utils_msg(self):

        for r in self._constraints:
            self._joined_utils = join_utils(self._joined_utils, r)

        # use projection to eliminate self out of the message to our parent
        util = projection(self._joined_utils, self._variable, self._mode)

        return util

    @register("VALUE")
    def _on_value_message(self, variable_name, recv_msg, t):
        self.logger.debug(f'{self.name}: on value message from {variable_name} : "{recv_msg}"')

        value = recv_msg.content
        msg_count, msg_size = 0, 0

        # Value msg contains the optimal assignment for all variables in our
        # separator : sep_vars, sep_values = value
        value_dict = {k.name: v for k, v in zip(*value)}
        self.logger.debug('Slicing relation on %s', value_dict)

        # as the value msg contains values for all variables in our
        # separator, slicing the util on these variables produces a relation
        # with a single dimension, our own variable.
        rel = self._joined_utils.slice(value_dict)

        self.logger.debug('Relation after slicing %s', rel)

        values, current_cost = find_arg_optimal(self._variable, rel, self._mode)
        selected_value = values[0]

        for c in self._children:
            variables_msg = [self._variable]
            values_msg = [selected_value]

            # own_separator intersection child_separator union
            # self.current_value
            for v in self._children_separator[c]:
                try:
                    values_msg.append(value_dict[v.name])
                    variables_msg.append(v)
                except KeyError:
                    # we want an intersection, we can ignore the variable if
                    # not in value_dict
                    pass
            msg = MBDpopMessage('VALUE', (variables_msg, values_msg))
            msg_count += 1
            msg_size += msg.size
            self.post_msg(c, msg)

        self.select_value_and_finish(selected_value, float(current_cost))

    """
    Labeling should be done before UTIL
    """

    @register("LABEL")
    def _on_label_message(self, variable_name, recv_msg, t):
        self.logger.debug(f'{self.name}: on label message from {variable_name} : "{recv_msg}"')

        label = recv_msg.content
        msg_count, msg_size = 0, 0

        # Value msg contains 2 list, one is the separator and
        # one is the list of CC nodes for its sender
        [recv_sep, recv_cc] = label

        # accumulate util messages until we got the LABEL from all our children
        self.union_seperators(recv_sep)
        self._cc_children.append(recv_cc)

        try:
            self._waited_children_label.remove(variable_name)
        except ValueError as e:
            self.logger.error(f'Unexpected LABEL message from {variable_name} on {self.name} : {recv_msg} ')
            raise e

        if len(self._waited_children_label) == 0:
            """
            Heuristic labelling of nodes as CC:
            If the separator of Sep_i of node X_i contains more than k nodes, then
            this ensures that enough of them will be labeled as cycle_cuts

            This part is done based on Algo 1: Labeling Protocol in the original paper

            TODO

            It is unclear if random pick is good enough or mechanism 1 and 2 are needed.

            """

            # get a union of all cc received from children
            cc_set = [set(x) for x in self._cc_children]
            cc_lists = set().union(*cc_set)

            if len(self._separators) <= k:

                if len(cc_lists) != 0:
                    self._label_CR="CR"
                    # the following step creates a idealogical cache table
                    # some function gets the domain of sep
                    # domains = [get_domain(sep) for sep in self._separators]
                    # TODO
                    # here I simplified the situation
                    # cache table will be { (val_sep1, val_sep2, ...): .}
                    domains = [ [x for x in self._variable.domain] for sep in self._separators]
                    domain_combination = list(itertools.product(*domains))
                    self._cache = { instantiation:"" for instantiation in domain_combination}
                else:
                    self._label_CR="normal"

                self._cc=[]

            else:

                # n are the nodes in sep_i but not marked as CC nodes by X_i's children
                N = set(self._separators) - cc_lists
                # select a new set of cycle cuts of length |N|-k
                # here I take the random way, should be updated to use mech1 or mech 2
                cc_new = set([x for x in N][:len(N) - k])
                self._cc = cc_lists | cc_new

            msg = MBDpopMessage('LABEL', [self._separators, self._cc])
            self.logger.info(f'On LABEL message from {variable_name}, \
                                send LABEL msg to parent {self._parent}')
            self.post_msg(self._parent, msg)
            msg_count += 1
            msg_size += msg.size


    def union_seperators(self, recv_sep):
        """
        Def:
        Ancestors of Xi which are direcly connected with Xi or descendants of Xi.

        Which can be easily determined by the union of
        a) seperators received from its children
        b) its parents and pp minus its self

        :return:

        a list of seperator nodes

        """

        self._separators = list(set(self._separators) | set(recv_sep))

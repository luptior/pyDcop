import logging
import sys
import threading
import traceback
import random
from functools import partial
from importlib import import_module
from threading import Thread
from time import perf_counter, sleep
from typing import Dict, List, Optional, Union, Callable, Tuple

from collections import defaultdict

from pydcop.algorithms import AlgorithmDef, ComputationDef, load_algorithm_module
from pydcop.dcop.objects import AgentDef, create_binary_variables
from pydcop.dcop.objects import BinaryVariable
from pydcop.dcop.relations import Constraint
from pydcop.infrastructure.Events import event_bus
from pydcop.infrastructure.communication import Messaging, \
    CommunicationLayer, UnreachableAgent
from pydcop.infrastructure.computations import MessagePassingComputation, \
    build_computation
from pydcop.infrastructure.discovery import Discovery, UnknownComputation, \
    UnknownAgent, _is_technical
from pydcop.infrastructure.ui import UiServer
from pydcop.reparation import create_computation_hosted_constraint, \
    create_agent_capacity_constraint, create_agent_hosting_constraint, \
    create_agent_comp_comm_constraint

from pydcop.infrastructure.agents import Agent


class Mailer_agent(Agent):
    """
    An agent which is the control piece of the

    Parameters
    ----------
    name: str
        name of the agent
    comm: CommunicationLayer
        object used to send and receive messages
    agent_def: AgentDef
        definition of this agent, optional
    ui_port: int
        the port on which to run the ui-server. If not given, no ui-server is
        started.
    delay: int
        An optional delay between message delivery, in second. This delay
        only applies to algorithm's messages and is useful when you want to
        observe (for example with the GUI) the behavior of the algorithm at
        runtime.

    delay_type: str
        either 'const', 'size'
        const is a constant number of delay, 'size' means the delay time is up to package size.
        In 'size' type, delay should be a function instead of number
    delay:
        either a function or a number
    loss_rate:
        there is a certain probability that a package can be lost in propagation
    message_queue:
        message received buy mailer will be FIFO

    """

    def __init__(self,
                 name,
                 comm: CommunicationLayer,
                 delay_type='const',
                 delay_p: float = 0,
                 loss_rate: float = 0,
                 agent_def: AgentDef = None,
                 ui_port: int = None,
                 delay: float = None):
        super().__init__(name, comm, agent_def, ui_port=ui_port, delay=delay)

        # custormized delay
        if not (delay_type == 'size' and callable(delay_p) or
                delay_type == 'const' and isinstance(delay_p, (int, float))):
            sys.exit("delay input is not consistent with delay type")

        self.delay_type = delay_type
        self.delay = delay
        self.loss_rate = loss_rate

import socket
import random
import sys
import queue
import pickle

import logging
import threading
import traceback
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


class Mailer:
    '''
    Mailer is the intermediate class in message propagation, that all agents first send its message to mailer, then mailer
    can decide when or whether send it to its original destination.

    instance:
    addr:
        will assign Mailer agent a static address
    comm:
        communication layer as defined in infrastructure
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

    '''

    def __init__(self,
                 addr,
                 comm: CommunicationLayer,
                 delay_type='const',
                 delay: float = 0,
                 loss_rate: float = 0):
        self.agent_name = 'Mailer'

        # communication part
        self.addr = addr
        self._comm = comm

        # custormized delay
        if not (delay_type == 'size' and callable(delay) or
                delay_type == 'const' and isinstance(delay, (int, float))):
            sys.exit("delay input is not consistent with delay type")

        self.delay_type = delay_type
        self.delay = delay
        self.loss_rate = loss_rate
        # set the queue to be an infinite queue
        self.message_queue = queue.Queue(0)

    def communication(self) -> CommunicationLayer:
        """
        The communication used by this agent.

        Returns
        -------
        CommunicationLayer
            The communication used by this agent.
        """
        return self._comm

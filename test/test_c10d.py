import copy
import math
import multiprocessing
import os
import random
import sys
import tempfile
import threading
import time
import unittest
from datetime import timedelta

from itertools import groupby
from functools import wraps
from collections import namedtuple

import torch
import common_utils as common
from torch import nn
import torch.nn.functional as F
import torch.distributed as c10d
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from common_utils import TestCase, load_tests, run_tests
from common_utils import retry_on_address_already_in_use_error

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if not c10d.is_available():
    print('c10d not available, skipping tests')
    sys.exit(0)


TIMEOUT_DEFAULT = 30
TIMEOUT_OVERRIDE = {}

TestSkip = namedtuple('TestSkip', 'exit_code, message')

TEST_SKIPS = {
    "multi-gpu": TestSkip(75, "Need at least 2 CUDA devices"),
    "nccl": TestSkip(76, "c10d not compiled with NCCL support"),
    "known_issues": TestSkip(77, "Test skipped due to known issues")
}


def skip_if_not_multigpu(func):
    """Multi-GPU tests requires at least 2 GPUS. Skip if this is not met."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            return func(*args, **kwargs)
        sys.exit(TEST_SKIPS['multi-gpu'].exit_code)

    return wrapper


def skip_if_lt_x_gpu(x):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            sys.exit(TEST_SKIPS['multi-gpu'].exit_code)
        return wrapper

    return decorator


def skip_if_not_nccl(func):
    """Skips a test if NCCL is not available (for c10d)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if hasattr(c10d, "ProcessGroupNCCL"):
            return func(*args, **kwargs)
        sys.exit(TEST_SKIPS['nccl'].exit_code)

    return wrapper


def skip_for_known_issues(func):
    """Skips a test due to known issues (for c10d)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        sys.exit(TEST_SKIPS['known_issues'].exit_code)

    return wrapper


def get_timeout(test_id):
    return TIMEOUT_OVERRIDE.get(test_id.split('.')[-1], TIMEOUT_DEFAULT)


def gpus_for_rank(world_size):
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    visible_devices = list(range(torch.cuda.device_count()))
    gpus_per_process = torch.cuda.device_count() // world_size
    gpus_for_rank = []
    for rank in range(world_size):
        gpus_for_rank.append(visible_devices[rank * gpus_per_process: (rank + 1) * gpus_per_process])
    return gpus_for_rank


def simple_reduce_tests(rank, world_size):
    return [
        (
            c10d.ReduceOp.SUM,
            torch.Tensor([rank + 1.0]),
            torch.Tensor([float(world_size * (world_size + 1) / 2)]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            torch.Tensor([rank + 1.0]),
            torch.Tensor([float(math.factorial(world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            torch.Tensor([rank + 1.0]),
            torch.Tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            torch.Tensor([rank + 1.0]),
            torch.Tensor([world_size]),
        ),
    ]


def simple_multi_input_reduce_tests(rank, world_size):
    return [
        (
            c10d.ReduceOp.SUM,
            [torch.Tensor([2 * rank + 0.0]), torch.Tensor([2 * rank + 1.0])],
            torch.Tensor([float(world_size * (2 * world_size - 1))]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            [torch.Tensor([2 * rank + 1.0]), torch.Tensor([2 * rank + 2.0])],
            torch.Tensor([float(math.factorial(2 * world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            [torch.Tensor([2 * rank + 1.0]), torch.Tensor([2 * rank + 2.0])],
            torch.Tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            [torch.Tensor([2 * rank + 1.0]), torch.Tensor([2 * rank + 2.0])],
            torch.Tensor([2 * world_size]),
        ),
    ]


class StoreTestBase(object):
    def _create_store(self, i):
        raise RuntimeError("not implemented")

    def _test_set_get(self, fs):
        fs.add("key", 1)
        fs.add("key", 2)
        fs.add("key", 3)
        fs.set("key0", "value0")
        fs.add("key3", 1)
        fs.set("key1", "value1")
        fs.add("key3", 2)
        fs.set("key2", "value2")
        fs.add("key3", 3)
        fs.add("key3", 4)
        fs.add("key3", 5)
        fs.add("key3", 6)
        self.assertEqual(b"6", fs.get("key"))
        self.assertEqual(b"value0", fs.get("key0"))
        self.assertEqual(b"value1", fs.get("key1"))
        self.assertEqual(b"value2", fs.get("key2"))
        self.assertEqual(b"21", fs.get("key3"))

    def test_set_get(self):
        self._test_set_get(self._create_store())


class FileStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        super(FileStoreTest, self).setUp()
        self.file = tempfile.NamedTemporaryFile(delete=False)

    def _create_store(self):
        store = c10d.FileStore(self.file.name, 1)
        store.set_timeout(timedelta(seconds=300))
        return store


class PrefixFileStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        super(PrefixFileStoreTest, self).setUp()
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.filestore = c10d.FileStore(self.file.name, 1)
        self.prefix = "test_prefix"
        self.filestore.set_timeout(timedelta(seconds=300))

    def _create_store(self):
        return c10d.PrefixStore(self.prefix, self.filestore)


def create_tcp_store(addr):
    """
    Creates a TCP store. Retries if the chosen port is already in use.
    """
    ports = []
    for _ in range(10):
        try:
            port = common.find_free_port()
            ports.append(port)
            return c10d.TCPStore(addr, port, 1, True)
        except RuntimeError as error:
            if str(error) == "Address already in use":
                continue
            raise
    raise RuntimeError("Unable to find free port (tried %s)" % ", ".join(ports))


class TCPStoreTest(TestCase, StoreTestBase):
    def _create_store(self):
        store = create_tcp_store('localhost')
        store.set_timeout(timedelta(seconds=300))
        return store

    def test_address_already_in_use(self):
        with self.assertRaisesRegex(RuntimeError, "^Address already in use$"):
            addr = 'localhost'
            port = common.find_free_port()

            # Use noqa to silence flake8.
            # Need to store in an unused variable here to ensure the first
            # object is not destroyed before the second object is created.
            store1 = c10d.TCPStore(addr, port, 1, True)  # noqa: F841
            store2 = c10d.TCPStore(addr, port, 1, True)  # noqa: F841


class PrefixTCPStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        super(PrefixTCPStoreTest, self).setUp()
        self.tcpstore = create_tcp_store('localhost')
        self.prefix = "test_prefix"
        self.tcpstore.set_timeout(timedelta(seconds=300))

    def _create_store(self):
        return c10d.PrefixStore(self.prefix, self.tcpstore)


class RendezvousTest(TestCase):
    def test_unknown_handler(self):
        with self.assertRaisesRegex(RuntimeError, "^No rendezvous handler"):
            c10d.rendezvous('invalid://')


class RendezvousEnvTest(TestCase):
    @retry_on_address_already_in_use_error
    def test_common_errors(self):
        # TODO remove this hack
        if not hasattr(c10d, "ProcessGroupNCCL"):
            raise unittest.SkipTest("C10D is not built with NCCL process group,"
                                    " skipping test")
        vars = {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": common.find_free_port(),
        }

        class Env(object):
            def __init__(self, vars):
                self.vars = vars

            def __enter__(self):
                for key, value in self.vars.items():
                    os.environ[key] = str(value)

            def __exit__(self, type, value, traceback):
                for key in self.vars.keys():
                    del os.environ[key]

        def without(d, key):
            d = d.copy()
            d.pop(key)
            return d

        def withouts(d, keys):
            d = d.copy()
            for key in keys:
                d.pop(key)
            return d

        with Env(without(vars, 'WORLD_SIZE')):
            with self.assertRaisesRegex(ValueError, 'WORLD_SIZE expected'):
                gen = c10d.rendezvous('env://')
                next(gen)
            c10d.init_process_group(backend='nccl', world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, 'RANK')):
            with self.assertRaisesRegex(ValueError, 'RANK expected'):
                gen = c10d.rendezvous('env://')
                next(gen)
            c10d.init_process_group(backend='nccl', rank=0)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(withouts(vars, ['RANK', 'WORLD_SIZE'])):
            c10d.init_process_group(backend='nccl', rank=0, world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(vars):
            c10d.init_process_group(backend='nccl')
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, 'MASTER_ADDR')):
            with self.assertRaisesRegex(ValueError, 'MASTER_ADDR expected'):
                gen = c10d.rendezvous('env://')
                next(gen)

        with Env(without(vars, 'MASTER_PORT')):
            with self.assertRaisesRegex(ValueError, 'MASTER_PORT expected'):
                gen = c10d.rendezvous('env://')
                next(gen)

        with Env(without(vars, 'WORLD_SIZE')):
            gen = c10d.rendezvous('env://?world_size={}'.format(1))
            _, _, size = next(gen)
            self.assertEqual(size, 1)

        with Env(without(vars, 'RANK')):
            gen = c10d.rendezvous('env://?rank={}'.format(0))
            _, rank, _ = next(gen)
            self.assertEqual(rank, 0)

        with Env(withouts(vars, ['RANK', 'WORLD_SIZE'])):
            gen = c10d.rendezvous('env://?rank={}&world_size={}'.format(0, 1))
            _, rank, size = next(gen)
            self.assertEqual(rank, 0)
            self.assertEqual(size, 1)

    @retry_on_address_already_in_use_error
    def test_nominal(self):
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(common.find_free_port())

        # Single rank
        os.environ['RANK'] = '0'
        gen0 = c10d.rendezvous('env://')
        store0, rank0, size0 = next(gen0)
        self.assertEqual(0, rank0)
        self.assertEqual(1, size0)

        store0.set("key0", "value0")

        # check with get
        self.assertEqual(b"value0", store0.get("key0"))


class RendezvousFileTest(TestCase):
    def test_common_errors(self):
        with self.assertRaisesRegex(ValueError, 'path missing'):
            gen = c10d.rendezvous('file://?rank=0&world_size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'rank parameter missing'):
            gen = c10d.rendezvous('file:///tmp/foo?world_size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'size parameter missing'):
            gen = c10d.rendezvous('file:///tmp/foo?rank=0')
            next(gen)

    def test_nominal(self):
        with tempfile.NamedTemporaryFile(delete=False) as file:
            url = 'file://%s?world_size=%d' % (file.name, 2)
            gen0 = c10d.rendezvous(url + "&rank=0")
            store0, rank0, size0 = next(gen0)
            self.assertEqual(0, rank0)
            self.assertEqual(2, size0)
            gen1 = c10d.rendezvous(url + "&rank=1")
            store1, rank1, size1 = next(gen1)
            self.assertEqual(1, rank1)
            self.assertEqual(2, size1)

            # Set value on both stores
            store0.set("key0", "value0")
            store1.set("key1", "value1")

            # Cross check with get
            self.assertEqual(b"value0", store1.get("key0"))
            self.assertEqual(b"value1", store0.get("key1"))


class RendezvousTCPTest(TestCase):
    def test_common_errors(self):
        with self.assertRaisesRegex(ValueError, 'port number missing'):
            gen = c10d.rendezvous('tcp://127.0.0.1?rank=0&world_size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'rank parameter missing'):
            gen = c10d.rendezvous('tcp://127.0.0.1:23456?world_size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'size parameter missing'):
            gen = c10d.rendezvous('tcp://127.0.0.1:23456?rank=0')
            next(gen)

    @retry_on_address_already_in_use_error
    def test_nominal(self):
        addr = 'localhost'
        port = common.find_free_port()
        url = 'tcp://%s:%d?world_size=%d' % (addr, port, 1)
        gen0 = c10d.rendezvous(url + "&rank=0")
        store0, rank0, size0 = next(gen0)
        self.assertEqual(0, rank0)
        self.assertEqual(1, size0)

        # Set value on the single store
        store0.set("key0", "value0")

        # check with get
        self.assertEqual(b"value0", store0.get("key0"))


class MultiProcessTestCase(TestCase):
    MAIN_PROCESS_RANK = -1

    @property
    def world_size(self):
        return 4

    @staticmethod
    def join_or_run(fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                fn(self)
        return wrapper

    # The main process spawns N subprocesses that run the test.
    # This function patches overwrites every test function to either
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    @classmethod
    def setUpClass(cls):
        for attr in dir(cls):
            if attr.startswith('test'):
                fn = getattr(cls, attr)
                setattr(cls, attr, cls.join_or_run(fn))

    def setUp(self):
        super(MultiProcessTestCase, self).setUp()
        self.rank = self.MAIN_PROCESS_RANK
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.processes = [self._spawn_process(rank) for rank in range(int(self.world_size))]

    def tearDown(self):
        super(MultiProcessTestCase, self).tearDown()
        for p in self.processes:
            p.terminate()

    def _spawn_process(self, rank):
        name = 'process ' + str(rank)
        process = multiprocessing.Process(target=self._run, name=name, args=(rank,))
        process.start()
        return process

    def _run(self, rank):
        self.rank = rank

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retreiving a corresponding test and executing it.
        getattr(self, self.id().split(".")[2])()
        sys.exit(0)

    def _join_processes(self, fn):
        timeout = get_timeout(self.id())
        start_time = time.time()
        for p in self.processes:
            p.join(timeout)
        elapsed_time = time.time() - start_time
        self._check_return_codes(elapsed_time)

    def _check_return_codes(self, elapsed_time):
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
        first_process = self.processes[0]
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError('Process {} terminated or timed out after {} seconds'.format(i, elapsed_time))
            self.assertEqual(p.exitcode, first_process.exitcode)
        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                raise unittest.SkipTest(skip.message)
        self.assertEqual(first_process.exitcode, 0)

    @property
    def is_master(self):
        return self.rank == 0


class TimeoutTest(TestCase):
    def _test_store_timeout(self, backend, init_method, c2p):
        try:
            c10d.distributed_c10d.init_process_group(
                backend=backend, init_method=init_method, world_size=1, rank=0,
                timeout=timedelta(seconds=1))
            default_store = c10d.distributed_c10d._get_default_store()
            tik = time.time()
            with self.assertRaisesRegex(RuntimeError, "Timeout"):
                default_store.get("nonexistent key")
            tok = time.time()
            c10d.destroy_process_group()
            c2p.append(float(tok - tik))
        except RuntimeError as e:
            # catch "Address already in use" error and report it to the main
            # thread
            c2p.append(e)

    def _init_methods(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        yield "file://%s" % f.name
        f.close()
        yield "tcp://127.0.0.1:%d" % common.find_free_port()

    def _test_default_store_timeout(self, backend):
        for init_method in self._init_methods():
            c2p = []
            t = threading.Thread(
                target=self._test_store_timeout,
                args=(backend, init_method, c2p))
            t.daemon = True
            t.start()
            t.join(5)

            self.assertEqual(1, len(c2p))
            if isinstance(c2p[0], float):
                # waiting time should be 1s, use 3s to rule out false alarm
                self.assertGreater(3, c2p[0])
            elif isinstance(c2p[0], RuntimeError):
                # let @retry_on_address_already_in_use_error handle the error
                raise c2p[0]
            else:
                raise RuntimeError("Unexpected type {}".format(type(c2p[0])))

    @retry_on_address_already_in_use_error
    def test_default_store_timeout_nccl(self):
        # TODO remove this hack
        if not hasattr(c10d, "ProcessGroupNCCL"):
            raise unittest.SkipTest("C10D is not built with NCCL process group,"
                                    " skipping test")
        self._test_default_store_timeout('nccl')

    @retry_on_address_already_in_use_error
    def test_default_store_timeout_gloo(self):
        self._test_default_store_timeout('gloo')


class ProcessGroupGlooTest(MultiProcessTestCase):
    def opts(self, threads=2):
        opts = c10d.ProcessGroupGloo.Options()
        opts.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        opts.timeout = 5.0
        opts.threads = threads
        return opts

    def test_broadcast_checks(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = -1
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t2], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t3], opts)

    def _test_broadcast_basics(self, fn):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # Every rank is root once
        for i in range(self.world_size):
            # Run with 1 input tensor
            x = fn(torch.Tensor([self.rank]))
            broadcast([x], i, 0)
            self.assertEqual(torch.Tensor([i]), x)

            # Run with 2 input tensors
            num = 2
            for j in range(num):
                xs = [
                    fn(torch.Tensor([self.rank * num + 0.0])),
                    fn(torch.Tensor([self.rank * num + 1.0])),
                ]

                broadcast(xs, i, j)
                self.assertEqual(torch.Tensor([i * num + j]), xs[0])
                self.assertEqual(torch.Tensor([i * num + j]), xs[1])

        # Test overloaded convenience function
        x = torch.Tensor([self.rank + 1.0])
        work = pg.broadcast(x, root=0)
        work.wait()
        self.assertEqual(torch.Tensor([1.0]), x)

    def test_broadcast_basics(self):
        self._test_broadcast_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_broadcast_basics_cuda(self):
        self._test_broadcast_basics(lambda t: t.clone().cuda())

    def _test_broadcast_stress(self, inputs):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts(threads=8))
        work_handles = [
            pg.broadcast(inputs[i], root=(i % self.world_size))
            for i in range(len(inputs))
        ]
        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(
                torch.Tensor([
                    (i * self.world_size) + (i % self.world_size)
                ]),
                inputs[i],
                "Mismatch in iteration %d" % i,
            )

    def test_broadcast_stress(self):
        inputs = [torch.Tensor([i * self.world_size + self.rank]) for i in range(1000)]
        self._test_broadcast_stress(inputs)

    @skip_if_not_multigpu
    def test_broadcast_stress_cuda(self):
        inputs = [torch.Tensor([i * self.world_size + self.rank]).cuda() for i in range(1000)]
        self._test_broadcast_stress(inputs)

    def test_allreduce_checks(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "requires non-empty tensor list"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t3], opts)

    def _test_allreduce_basics(self, fn):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Single input tests
        tests = simple_reduce_tests(self.rank, self.world_size)
        for (op, input, output) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input)
            work = pg.allreduce([tensor], opts)
            work.wait()
            self.assertEqual(output, tensor)

        # Multi input tests
        tests = simple_multi_input_reduce_tests(self.rank, self.world_size)
        for (op, inputs, output) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensors = [fn(input) for input in inputs]
            work = pg.allreduce(tensors, opts)
            work.wait()
            for tensor in tensors:
                self.assertEqual(output, tensor)

        # Test overloaded convenience function (defaults to using sum)
        x = fn(torch.Tensor([self.rank + 1.0]))
        work = pg.allreduce(x)
        work.wait()
        self.assertEqual(torch.Tensor([float(self.world_size * (self.world_size + 1) / 2)]), x)

    def test_allreduce_basics(self):
        self._test_allreduce_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_allreduce_basics_cuda(self):
        self._test_allreduce_basics(lambda t: t.clone().cuda())

    def _test_allreduce_stress(self, inputs):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts(threads=8))
        work_handles = [pg.allreduce(inputs[i]) for i in range(len(inputs))]
        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(
                torch.Tensor([
                    (i * self.world_size) +
                    (self.world_size * (self.world_size - 1) / 2)
                ]),
                inputs[i],
                "Mismatch in iteration %d" % i,
            )

    def test_allreduce_stress(self):
        inputs = [torch.Tensor([i + self.rank]) for i in range(1000)]
        self._test_allreduce_stress(inputs)

    @skip_if_not_multigpu
    def test_allreduce_stress_cuda(self):
        inputs = [torch.Tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_allreduce_stress(inputs)

    def test_scatter_checks(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = -1
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.world_size
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element output tensor list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element output tensor list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([t1, t1], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * self.world_size, [t1] * self.world_size], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * (self.world_size - 1)], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * (self.world_size + 1)], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * (self.world_size + 1)], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t2] * self.world_size], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t3] * self.world_size], opts)

        with self.assertRaisesRegex(ValueError, "requires empty input on non-root"):
            opts = c10d.ScatterOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.scatter([t1], [[t1] * self.world_size], opts)

    def _test_scatter_basics(self, fn):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Preallocate tensors for input/output
        input = [fn(torch.Tensor([self.rank])) for _ in range(self.world_size)]
        outputs = [fn(torch.Tensor([-1])) for _ in range(self.world_size)]

        # Take turns being the scatter root and accumulate work items
        work = []
        for i in range(self.world_size):
            opts = c10d.ScatterOptions()
            opts.rootRank = i
            if i == self.rank:
                work.append(pg.scatter([outputs[i]], [input], opts))
            else:
                work.append(pg.scatter([outputs[i]], [], opts))

        # Wait for work to complete
        for i in range(self.world_size):
            work[i].wait()
            self.assertEqual(torch.Tensor([i]), outputs[i])

    def test_scatter_basics(self):
        self._test_scatter_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_scatter_basics_cuda(self):
        self._test_scatter_basics(lambda t: t.clone().cuda())

    def _test_scatter_stress(self, inputs, fn):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts(threads=8))
        outputs = [
            [fn(torch.Tensor([-1])) for _ in range(self.world_size)]
            for _ in range(len(inputs))
        ]
        work_handles = []
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.ScatterOptions()
                opts.rootRank = root
                if root == self.rank:
                    work = pg.scatter([outputs[i][root]], [[fn(e) for e in inputs[i]]], opts)
                else:
                    work = pg.scatter([outputs[i][root]], [], opts)
                work_handles.append(work)

        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size

            self.assertEqual(
                torch.Tensor([iter + root]),
                outputs[iter][root],
                "Mismatch in iteration %d for rank %d" % (iter, root)
            )

    def test_scatter_stress(self):
        inputs = [
            [torch.Tensor([i + self.rank]) for _ in range(self.world_size)]
            for i in range(1000)
        ]
        self._test_scatter_stress(inputs, lambda t: t.clone())

    @unittest.skip("Test is flaky, see https://github.com/pytorch/pytorch/issues/15963")
    @skip_if_not_multigpu
    def test_scatter_stress_cuda(self):
        inputs = [
            [torch.Tensor([i + self.rank]) for _ in range(self.world_size)]
            for i in range(1000)
        ]
        self._test_scatter_stress(inputs, lambda t: t.clone().cuda())

    def test_gather_checks(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = -1
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.world_size
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input tensor list"):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input tensor list"):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [t1, t1], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element output list"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element output list"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * self.world_size, [t1] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element output list"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * (self.world_size - 1)], [t1], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element output list"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * (self.world_size + 1)], [t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t2] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t3] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(ValueError, "requires empty output on non-root"):
            opts = c10d.GatherOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.gather([[t1] * self.world_size], [t1], opts)

    def _test_gather_basics(self, fn):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Preallocate tensors for input/output
        input = [fn(torch.Tensor([self.rank]))]
        outputs = [fn(torch.Tensor([-1])) for _ in range(self.world_size)]

        # Take turns being the gather root and accumulate work items
        work = []
        for i in range(self.world_size):
            opts = c10d.GatherOptions()
            opts.rootRank = i
            if i == self.rank:
                work.append(pg.gather([outputs], input, opts))
            else:
                work.append(pg.gather([], input, opts))

        # Wait for work to complete
        expected = [torch.Tensor([rank]) for rank in range(self.world_size)]
        for i in range(self.world_size):
            work[i].wait()
            if i == self.rank:
                self.assertEqual(expected, outputs)

    def test_gather_basics(self):
        self._test_gather_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_gather_basics_cuda(self):
        self._test_gather_basics(lambda t: t.clone().cuda())

    def _test_gather_stress(self, inputs, fn):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts(threads=8))
        work_handles = []
        outputs = [
            [
                [fn(torch.Tensor([-1])) for _ in range(self.world_size)]
            ] for _ in range(len(inputs))
        ]
        expected_outputs = [
            [
                [torch.Tensor([i + j]) for j in range(self.world_size)]
            ] for i in range(len(inputs))
        ]
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.GatherOptions()
                opts.rootRank = root
                if root == self.rank:
                    work = pg.gather(outputs[i], [fn(inputs[i])], opts)
                else:
                    work = pg.gather([], [fn(inputs[i])], opts)
                work_handles.append(work)

        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size
            if root == self.rank:
                self.assertEqual(
                    expected_outputs[iter],
                    outputs[iter],
                    "Mismatch in iteration %d for root %d" % (iter, root)
                )

    def test_gather_stress(self):
        inputs = [torch.Tensor([i + self.rank]) for i in range(1000)]
        self._test_gather_stress(inputs, lambda t: t.clone())

    @skip_if_not_multigpu
    def test_gather_stress_cuda(self):
        inputs = [torch.Tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_gather_stress(inputs, lambda t: t.clone().cuda())

    def test_allgather_checks(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "requires non-empty input tensor list"):
            pg.allgather([], [])

        with self.assertRaisesRegex(ValueError, "requires input/output tensor lists to have the same length"):
            pg.allgather([], [t1])

        with self.assertRaisesRegex(ValueError, "requires input/output tensor lists to have the same length"):
            pg.allgather([[t1] * self.world_size, [t1] * self.world_size], [t1])

        with self.assertRaisesRegex(ValueError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size - 1)], [t1])

        with self.assertRaisesRegex(ValueError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size + 1)], [t1])

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            pg.allgather([[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t2])

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            pg.allgather([[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t3])

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            pg.allgather([([t1, t2] * (self.world_size))[:self.world_size]], [t1])

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            pg.allgather([([t1, t3] * (self.world_size))[:self.world_size]], [t1])

    def _test_allgather_basics(self, fn):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Run with N input tensor per rank
        for n in [1, 2, 3]:
            input = [
                fn(torch.Tensor([n * self.rank + i])) for i in range(n)
            ]
            output = [
                [
                    fn(torch.Tensor([-1])) for _ in range(n * self.world_size)
                ] for _ in range(n)
            ]
            expected_output = [
                [
                    torch.Tensor([i]) for i in range(n * self.world_size)
                ] for _ in range(n)
            ]
            work = pg.allgather(output, input)
            work.wait()
            self.assertEqual(expected_output, output)

    def test_allgather_basics(self):
        self._test_allgather_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_allgather_basics_cuda(self):
        self._test_allgather_basics(lambda t: t.clone().cuda())

    def _test_allgather_stress(self, inputs, fn):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts(threads=8))
        work_handles = []
        outputs = [
            [
                [fn(torch.Tensor([-1])) for _ in range(self.world_size)]
            ] for _ in range(len(inputs))
        ]
        expected_outputs = [
            [
                [torch.Tensor([i + j]) for j in range(self.world_size)]
            ] for i in range(len(inputs))
        ]
        for i in range(len(inputs)):
            work = pg.allgather(outputs[i], [fn(inputs[i])])
            work_handles.append(work)

        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(
                expected_outputs[i],
                outputs[i],
                "Mismatch in iteration %d" % i
            )

    def test_allgather_stress(self):
        inputs = [torch.Tensor([i + self.rank]) for i in range(1000)]
        self._test_allgather_stress(inputs, lambda t: t.clone())

    @skip_if_not_multigpu
    def test_allgather_stress_cuda(self):
        inputs = [torch.Tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_allgather_stress(inputs, lambda t: t.clone().cuda())

    def test_reduce_checks(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ReduceOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element tensor list"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.reduce([t1, t1], opts)

    def _test_reduce_basics(self, fn):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())
        for (op, input, output) in simple_reduce_tests(self.rank, self.world_size):
            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.reduceOp = op
                opts.rootRank = root
                tmp = fn(input)
                work = pg.reduce([tmp], opts)
                work.wait()
                if root == self.rank:
                    self.assertEqual(output, tmp)

    def test_reduce_basics(self):
        self._test_reduce_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_reduce_basics_cuda(self):
        self._test_reduce_basics(lambda t: t.clone().cuda())

    def _test_reduce_stress(self, inputs):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts(threads=8))
        work_handles = []
        outputs = []
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.rootRank = root
                tmp = inputs[i].clone()
                outputs.append(tmp)
                work = pg.reduce([tmp], opts)
                work_handles.append(work)

        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size
            if root == self.rank:
                self.assertEqual(
                    torch.Tensor([
                        (iter * self.world_size) +
                        (self.world_size * (self.world_size - 1) / 2)
                    ]),
                    outputs[i],
                    "Mismatch in iteration %d with root rank %d" % (iter, root),
                )

    def test_reduce_stress(self):
        inputs = [torch.Tensor([i + self.rank]) for i in range(1000)]
        self._test_reduce_stress(inputs)

    @skip_if_not_multigpu
    def test_reduce_stress_cuda(self):
        inputs = [torch.Tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_reduce_stress(inputs)

    def test_send_recv_all_to_all(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Preallocate tensors for input/output
        inputs = [torch.Tensor([self.rank]) for _ in range(self.world_size)]
        outputs = [torch.Tensor([-1]) for _ in range(self.world_size)]

        # Issue sends
        send_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            send_work.append(pg.send([inputs[i]], i, 0))

        # Issue recvs
        recv_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            recv_work.append(pg.recv([outputs[i]], i, 0))

        # Wait for sends to complete
        for work in send_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # Wait for recvs to complete
        for work in recv_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # Test that every output other than our own contains the respective rank
        for i in range(self.world_size):
            if i == self.rank:
                continue
            self.assertEqual(torch.Tensor([i]), outputs[i])

    def test_timeout_kwarg(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=0.5))

        # Wait on barrier
        pg.barrier().wait()

        # Sleep on one of the processes to trigger barrier timeout
        if self.rank == 0:
            time.sleep(1.0)

        # The barrier will now time out
        with self.assertRaisesRegex(RuntimeError, " (Timed out|closed) "):
            pg.barrier().wait()

    def test_barrier_implies_wait(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Kick off allreduce operations
        size = (100, 100)
        num = 16
        tensors = [torch.full(size, float(i)) for i in range(num)]
        for tensor in tensors:
            # Note: leak the returned work handle
            pg.allreduce(tensor)

        # Barrier should ensure all previous work has completed
        pg.barrier().wait()

        for i, tensor in enumerate(tensors):
            self.assertEqual(torch.full(size, float(i * self.world_size)), tensor)


class ProcessGroupNCCLTest(TestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        if not hasattr(c10d, "ProcessGroupNCCL"):
            raise unittest.SkipTest("C10D is not built with NCCL process group,"
                                    " skipping test")

        self.rank = self.MAIN_PROCESS_RANK
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            raise unittest.SkipTest("NCCL test requires 2+ GPUs")

    def tearDown(self):
        pass

    def test_broadcast_ops(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # for every root tensor
        for rt in range(self.num_gpus):
            tensors = []
            for i in range(self.num_gpus):
                tensors.append(torch.Tensor([i]).cuda(i))

            broadcast(tensors, self.rank, rt)

            for i in range(self.num_gpus):
                self.assertEqual(tensors[i], tensors[rt])

    def test_allreduce_ops(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def allreduce(tensors, op):
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            work = pg.allreduce(tensors, opts)
            work.wait()

        # Sum
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.SUM)

        for i in range(self.num_gpus):
            self.assertEqual(
                torch.Tensor([float(self.num_gpus * (self.num_gpus + 1) / 2)]),
                tensors[i])

        # Product
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.PRODUCT)

        for i in range(self.num_gpus):
            self.assertEqual(
                torch.Tensor([float(math.factorial(self.num_gpus))]),
                tensors[i])

        # Min
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.MIN)

        for i in range(self.num_gpus):
            self.assertEqual(torch.Tensor([1.0]), tensors[i])

        # Max
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.MAX)

        for i in range(self.num_gpus):
            self.assertEqual(torch.Tensor([self.num_gpus]), tensors[i])

    def test_reduce_ops(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def reduce(xs, rootRank, rootTensor):
            opts = c10d.ReduceOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.reduce(xs, opts)
            work.wait()

        # for every root tensor
        for rt in range(self.num_gpus):
            tensors = []
            for i in range(self.num_gpus):
                tensors.append(torch.Tensor([i + 1]).cuda(i))

            reduce(tensors, self.rank, rt)

            self.assertEqual(
                torch.Tensor([float(self.num_gpus * (self.num_gpus + 1) / 2)]),
                tensors[rt])

    def test_allgather_ops(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def allgather(output_ts, input_ts):
            work = pg.allgather(output_ts, input_ts)
            work.wait()

        tensors = []
        output_ts = [[] for _ in range(self.num_gpus)]

        for idx, ls in enumerate(output_ts):
            for _ in range(self.world_size * self.num_gpus):
                ls.append(torch.Tensor([0]).cuda(idx))

        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i]).cuda(i))

        allgather(output_ts, tensors)

        # Verification
        for device_ts in output_ts:
            for s_idx, t in enumerate(device_ts):
                self.assertEqual(torch.Tensor([s_idx]), t)

    def test_reduce_scatter_ops(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def reduce_scatter(outputs, input_lists, op):
            opts = c10d.ReduceScatterOptions()
            opts.reduceOp = op
            work = pg.reduce_scatter(outputs, input_lists, opts)
            work.wait()

        virtual_rank = self.rank * self.world_size
        virtual_world_size = self.num_gpus * self.world_size

        output = [
            torch.Tensor([0]).cuda(i)
            for i in range(self.num_gpus)
        ]

        #           0                   1                   2
        #   0   [0..11]             [1..12]
        #   1   [3..14]
        #   2
        #   3

        # Sum
        tensor_lists = [
            [
                torch.Tensor([self.rank * self.num_gpus + i + j]).cuda(i)
                for j in range(virtual_world_size)
            ]
            for i in range(self.num_gpus)
        ]

        reduce_scatter(output, tensor_lists, c10d.ReduceOp.SUM)

        for i in range(self.num_gpus):
            expected = torch.Tensor([
                float(self.num_gpus * (self.num_gpus - 1) / 2) +
                (virtual_rank + i) * virtual_world_size
            ])
            self.assertEqual(expected, output[i])

        # Min
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.MIN)

        for i in range(self.num_gpus):
            expected = torch.Tensor([self.rank * self.world_size + i])
            self.assertEqual(expected, output[i])

        # Max
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.MAX)

        for i in range(self.num_gpus):
            expected = torch.Tensor(
                [self.rank * self.world_size + i + virtual_world_size - 1]
            )
            self.assertEqual(expected, output[i])

        # Product
        tensor_lists = [
            [
                torch.Tensor([
                    (self.rank * self.num_gpus + i + j) % virtual_world_size + 1
                ]).cuda(i)
                for j in range(virtual_world_size)
            ]
            for i in range(self.num_gpus)
        ]

        reduce_scatter(output, tensor_lists, c10d.ReduceOp.PRODUCT)

        for i in range(self.num_gpus):
            expected = torch.Tensor([float(math.factorial(virtual_world_size))])
            self.assertEqual(expected, output[i])

    def test_barrier(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def allreduce(tensors):
            opts = c10d.AllreduceOptions()
            work = pg.allreduce(tensors, opts)
            return work

        # Making the collective to operate on
        # 1, 2, 3, 4, .... self.num_gpus GPUs
        tensors_list = [[] for _ in range(2, self.num_gpus + 1)]
        for i in range(2, self.num_gpus + 1):
            for j in range(i):
                tensors_list[i - 2].append(torch.Tensor([j + 1]).cuda(j))

        works = []
        for tensors in tensors_list:
            work = allreduce(tensors)
            works.append(work)

        # Barrier will ensure that all previous work is completed
        pg.barrier().wait()

        for i in range(2, self.num_gpus + 1):
            for j in range(i):
                self.assertEqual(
                    torch.Tensor([float(i * (i + 1) / 2)]),
                    tensors_list[i - 2][j])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class DoubleGpuNet(nn.Module):
    def __init__(self, gpus):
        super(DoubleGpuNet, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False).to(gpus[0])
        self.fc2 = nn.Linear(10, 50, bias=False).to(gpus[1])
        self.fc3 = nn.Linear(50, 4, bias=False).to(gpus[1])
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(torch.Tensor([2, 2]).long(),
                                          requires_grad=False).to(gpus[0])

    def forward(self, x):
        dev0 = self.fc1.weight.device
        dev1 = self.fc2.weight.device
        x = self.relu(self.fc1(x.to(dev0)))
        x = self.relu(self.fc2(x.to(dev1)))
        x = self.fc3(x)
        return F.softmax(x, dim=1).to(dev0)


class QuadraGpuNet(nn.Module):
    def __init__(self, gpus):
        super(QuadraGpuNet, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False).to(gpus[0])
        self.fc2 = nn.Linear(10, 50, bias=False).to(gpus[1])
        self.fc3 = nn.Linear(50, 4, bias=False).to(gpus[2])
        self.fc4 = nn.Linear(4, 4, bias=False).to(gpus[3])
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(torch.Tensor([2, 2]).long(),
                                          requires_grad=False).to(gpus[0])

    def forward(self, x):
        dev0 = self.fc1.weight.device
        dev1 = self.fc2.weight.device
        dev2 = self.fc3.weight.device
        dev3 = self.fc4.weight.device
        x = self.relu(self.fc1(x.to(dev0)))
        x = self.relu(self.fc2(x.to(dev1)))
        x = self.relu(self.fc3(x.to(dev2)))
        x = self.fc4(x.to(dev3))
        return F.softmax(x, dim=1).to(dev0)


class DistributedDataParallelTest(MultiProcessTestCase):

    def tearDown(self):
        # DistributedDataParallel test doesn't seem to call FileStore destructor
        # TODO: investigate this test and the test is known to have issues
        # Use this hack to remove files for that test
        try:
            os.remove(self.file.name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    def _prepare_single_device_module(self, process_group, gpus, global_batch_size):
        model = Net()
        ddp_model = DistributedDataParallel(
            copy.deepcopy(model).cuda(gpus[0]),
            device_ids=gpus,
            process_group=process_group,
            bucket_cap_mb=0.001)

        model.cuda(gpus[0])

        input = torch.randn(global_batch_size, 2).cuda(gpus[0])
        target = torch.randn(global_batch_size, 4).cuda(gpus[0])

        return model, ddp_model, input, target

    def _prepare_multi_device_module(self, process_group, gpus, global_batch_size):
        self.assertTrue(
            len(gpus) == 2 or len(gpus) == 4,
            "unexpected devices for ddp tests {}".format(gpus))
        if len(gpus) == 2:
            model = DoubleGpuNet(gpus)
        elif len(gpus) == 4:
            model = QuadraGpuNet(gpus)

        ddp_model = DistributedDataParallel(
            copy.deepcopy(model),
            process_group=process_group,
            bucket_cap_mb=0.001)

        input = torch.randn(global_batch_size, 2).to(gpus[0])
        target = torch.randn(global_batch_size, 4)

        return model, ddp_model, input, target

    def _test_ddp_with_process_group(self, process_group, gpus, multi_gpu=False):
        local_batch_size = len(gpus)
        global_batch_size = self.world_size * local_batch_size

        if multi_gpu:
            model, ddp_model, input, target = \
                self._prepare_multi_device_module(
                    process_group, gpus, global_batch_size)
        else:
            model, ddp_model, input, target = \
                self._prepare_single_device_module(
                    process_group, gpus, global_batch_size)

        def step_model(model, input, target):
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()

        def update_parameters(model):
            for param in model.parameters():
                param.data -= param.grad
                param.grad = None

        # check two model parameters over 2 iterations
        for iteration in range(2):
            # single cpu/gpu training
            step_model(model, input, target)

            # DDP training, DDP scatters subsets of input_cpu to nodes/GPUs
            step_model(ddp_model,
                       input[self.rank * local_batch_size: (self.rank + 1) * local_batch_size],
                       target[self.rank * local_batch_size: (self.rank + 1) * local_batch_size])

            # Update weights and run a second iteration to shake out errors
            update_parameters(model)
            update_parameters(ddp_model)
            self.assertEqual(len(list(model.parameters())), len(list(ddp_model.parameters())))
            for i, j in zip(model.parameters(), ddp_model.parameters()):
                self.assertEqual(i, j)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    def _test_gloo_backend(self, gpus, multi_gpu=False, use_str=False):
        if use_str:
            gpus = list(map(lambda i: torch.device('cuda:' + str(i)), gpus))
        store = c10d.FileStore(self.file.name, self.world_size)
        options = c10d.ProcessGroupGloo.Options()
        options.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size, options)
        self._test_ddp_with_process_group(process_group, gpus, multi_gpu)

    @skip_if_not_multigpu
    def test_gloo_backend(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_gloo_backend(gpus)

    @skip_if_not_multigpu
    def test_gloo_backend_str(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_gloo_backend(gpus, use_str=True)

    @skip_if_lt_x_gpu(4)
    def test_gloo_backend_2gpu_module(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_gloo_backend(gpus[:2], multi_gpu=True)

    @skip_if_lt_x_gpu(4)
    def test_gloo_backend_2gpu_module_str(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_gloo_backend(gpus[:2], multi_gpu=True, use_str=True)

    @skip_if_lt_x_gpu(8)
    def test_gloo_backend_4gpu_module(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_gloo_backend(gpus[:4], multi_gpu=True)

    @skip_if_lt_x_gpu(8)
    def test_gloo_backend_4gpu_module_str(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_gloo_backend(gpus[:4], multi_gpu=True, use_str=True)

    def _test_nccl_backend(self, gpus, multi_gpu=False, use_str=False):
        if use_str:
            gpus = list(map(lambda i: torch.device('cuda:' + str(i)), gpus))
        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        self._test_ddp_with_process_group(process_group, gpus, multi_gpu)

    @skip_if_not_multigpu
    @skip_if_not_nccl
    def test_nccl_backend(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_nccl_backend(gpus)

    @skip_if_not_multigpu
    @skip_if_not_nccl
    def test_nccl_backend_str(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_nccl_backend(gpus, use_str=True)

    @skip_if_lt_x_gpu(4)
    @skip_if_not_nccl
    def test_nccl_backend_2gpu_module(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_nccl_backend(gpus[:2], multi_gpu=True)

    @skip_if_lt_x_gpu(4)
    @skip_if_not_nccl
    def test_nccl_backend_2gpu_module_str(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_nccl_backend(gpus[:2], multi_gpu=True, use_str=True)

    @skip_if_lt_x_gpu(8)
    @skip_if_not_nccl
    def test_nccl_backend_4gpu_module(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_nccl_backend(gpus[:4], multi_gpu=True)

    @skip_if_lt_x_gpu(8)
    @skip_if_not_nccl
    def test_nccl_backend_4gpu_module_str(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]
        self._test_nccl_backend(gpus[:4], multi_gpu=True, use_str=True)

    @skip_if_lt_x_gpu(4)
    @skip_if_not_nccl
    def test_ddp_multi_device_module_config(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]

        self.assertTrue(len(gpus) >= 2, "expecting at least 2 gpus per process")

        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        gpus = gpus[:2]
        model = DoubleGpuNet(gpus)

        with self.assertRaisesRegex(AssertionError, "output_device .* single-device CUDA"):
            ddp_model = DistributedDataParallel(
                model, output_device=gpus[1], process_group=process_group)

        with self.assertRaisesRegex(AssertionError, "device_ids .* single-device CUDA"):
            ddp_model = DistributedDataParallel(
                model, device_ids=gpus, process_group=process_group)

        with self.assertRaisesRegex(AssertionError, "only works with CUDA devices"):
            model.fc1 = model.fc1.cpu()
            ddp_model = DistributedDataParallel(model, process_group=process_group)

        model = model.cpu()
        with self.assertRaisesRegex(AssertionError, "device_ids .* single-device CUDA"):
            ddp_model = DistributedDataParallel(
                model, device_ids=gpus, process_group=process_group)

    @skip_if_not_multigpu
    @skip_if_not_nccl
    @skip_for_known_issues
    def test_dist_broadcast_coalesced_nccl(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        device = torch.device('cuda')

        for fine_grained in [False, True]:
            target = torch.arange(60, dtype=torch.float16, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float16, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float64, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float16, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)

            if self.is_master:
                # All processes should have these tensors in the end.
                tensors = target
            else:
                # Non-master processes start with empty tensors and should be
                # filled with the tensors from the master.
                tensors = torch.zeros(60, dtype=torch.float16, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float32, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float16, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float64, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float16, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float32, device=device).chunk(5)

            c10d._dist_broadcast_coalesced(
                process_group,
                tensors,
                buffer_size=256,
                fine_grained=fine_grained)

            self.assertEqual(tensors, target)

    @skip_if_not_multigpu
    def test_dist_broadcast_coalesced_gloo(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        options = c10d.ProcessGroupGloo.Options()
        options.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size, options)

        device = torch.device('cuda')

        for fine_grained in [False, True]:
            target = torch.arange(60, dtype=torch.float16, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float16, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float64, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float16, device=device).chunk(5)
            target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)

            if self.is_master:
                # All processes should have these tensors in the end.
                tensors = target
            else:
                # Non-master processes start with empty tensors and should be
                # filled with the tensors from the master.
                tensors = torch.zeros(60, dtype=torch.float16, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float32, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float16, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float64, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float16, device=device).chunk(5)
                tensors += torch.zeros(60, dtype=torch.float32, device=device).chunk(5)

            c10d._dist_broadcast_coalesced(
                process_group,
                tensors,
                buffer_size=128,
                fine_grained=fine_grained)

            self.assertEqual(tensors, target)

    @skip_if_not_multigpu
    def test_sync_params_no_buffers(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        options = c10d.ProcessGroupGloo.Options()
        options.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size, options)

        # Use all available devices on every process here (data is small, so should be fine).
        devices = gpus_for_rank(self.world_size)[self.rank]
        target = torch.arange(10, dtype=torch.float64, device='cuda:{}'.format(devices[0])).chunk(5)
        parameter_data = [target]
        parameter_data += [torch.zeros(10, device=torch.device('cuda', d)).chunk(5) for d in devices[1:]]
        buffer_data = [[]] * len(parameter_data)

        c10d._sync_params(
            process_group,
            parameter_data=parameter_data,
            buffer_data=buffer_data,
            devices=devices,
            broadcast_bucket_size=10,
            broadcast_buffers=False)

        for device_data in parameter_data:
            for i, parameter in enumerate(device_data):
                self.assertEqual(parameter, target[i])

    @skip_if_not_multigpu
    def test_sync_params_with_buffers(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        options = c10d.ProcessGroupGloo.Options()
        options.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size, options)

        devices = gpus_for_rank(self.world_size)[self.rank]
        target = torch.arange(10, dtype=torch.float64, device='cuda:{}'.format(devices[0])).chunk(5)
        parameter_data = [target]
        parameter_data += [torch.zeros(10, device=torch.device('cuda', d)).chunk(5) for d in devices[1:]]

        # sync_params should do a dist_broadcast for buffers, so we only populate the master buffers and
        # then check that other processes' tensors end up matching.

        if self.is_master:
            buffer_data = [target]
            buffer_data += [torch.zeros(10, device=torch.device('cuda', d)).chunk(5) for d in devices[1:]]
        else:
            buffer_data = [torch.zeros(10, device=torch.device('cuda', d)).chunk(5) for d in devices]

        c10d._sync_params(
            process_group,
            parameter_data=parameter_data,
            buffer_data=buffer_data,
            devices=devices,
            broadcast_bucket_size=10,
            broadcast_buffers=True)

        for device_data in parameter_data:
            for i, parameter in enumerate(device_data):
                self.assertEqual(parameter, target[i])

        for device_data in buffer_data:
            for i, buffer in enumerate(device_data):
                self.assertEqual(buffer, target[i])

    @skip_if_not_multigpu
    @skip_if_not_nccl
    def test_fp16(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        gpus = gpus_for_rank(self.world_size)[self.rank]
        model = nn.Linear(1, 1, bias=False).cuda(gpus[0]).half()
        nn.init.constant_(model.weight, 1)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[gpus[0]],
            process_group=process_group,
            bucket_cap_mb=0.001,
        )

        # Input 2**15, so that the gradients will overflow with a
        # world_size of 2, unless we normalize the gradient by the
        # world_size before the reduction
        input = torch.Tensor([[2**15]]).cuda(gpus[0]).half()

        # Step model
        ddp_model.train()
        output = ddp_model(input)
        loss = output.sum()
        loss.backward()

        self.assertFalse(
            any(torch.isinf(p.grad).any() for p in ddp_model.parameters())
        )

    @skip_if_not_nccl
    @skip_if_not_multigpu
    def test_queue_reduction(self):
        # Set up process group.
        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # Get this process' split of devices.
        devices = gpus_for_rank(self.world_size)[self.rank]
        grads_batch = [(torch.ones(10, device=torch.device('cuda', d)) *
                       (self.rank + 1)).chunk(5)
                       for d in devices]

        work, local_grad_sum = c10d._queue_reduction(process_group,
                                                     grads_batch,
                                                     devices)
        # The first return value should be the allreduce work item.
        self.assertTrue(isinstance(work, c10d.Work))
        # The second return value will be the finished allreduced gradients.
        self.assertTrue(isinstance(local_grad_sum, torch.Tensor))

        # Wait for the allreduce to finish.
        work.wait()

        # The expected result of the allreduce should be the average
        self.assertEqual(local_grad_sum,
                         torch.ones(10) * (self.world_size + 1) * len(devices) / 2.0)

    @skip_if_not_nccl
    @skip_if_not_multigpu
    def test_sync_reduction(self):
        # Set up process group.
        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # Get this process' split of devices.
        devices = gpus_for_rank(self.world_size)[self.rank]
        grads_batch = [(torch.ones(10, device=torch.device('cuda', d)) *
                       (self.rank + 1)).chunk(5)
                       for d in devices]
        work, local_grad_sum = c10d._queue_reduction(process_group,
                                                     grads_batch,
                                                     devices)
        c10d._sync_reduction(work, grads_batch[0], local_grad_sum)
        # The expected result of the allreduce should be the average
        self.assertEqual(grads_batch[0], (torch.ones(10) * (self.world_size + 1) * len(devices) / 2.0).chunk(5))

    @skip_if_not_nccl
    @skip_if_not_multigpu
    def test_arbitrary_forward_return_value(self):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        class ForwardReturnValueModule(nn.Module):
            def __init__(self):
                super(ForwardReturnValueModule, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x, fn):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # The first softmax does NOT include fc3 in its autograd graph
                # whereas the second softmax DOES. If we pass only the first
                # tensor we see in the output to the reducer, it marks the
                # gradient for fc3 as ready (because it doesn't show up). If
                # downstream uses of this return value choose to differentiate
                # against the second output tensor, it would still receive a
                # gradient and a callback for this tensor, resulting in a crash.
                return fn(
                    F.softmax(x, dim=1),
                    F.softmax(self.fc3(x), dim=1),
                )

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            ForwardReturnValueModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(device_id)

        # Always run "backward" to ensure the reducer is called by autograd.
        # If we don't correctly capture the output tensors from the return value,
        # the reducer won't see a hook for the unused parameter, and throw an error.
        # The correct capture is what we're testing in this function.
        def test(box, unbox):
            output = model(input, fn=box)
            loss = criterion(unbox(output), target)
            loss.backward()

        # Test with identity return value
        test(
            box=lambda x, y: (x, y),
            unbox=lambda obj: obj[1],
        )

        # Test with list return value
        test(
            box=lambda x, y: ["foo", x, "bar", y],
            unbox=lambda obj: obj[3],
        )

        # Test with tuple return value
        test(
            box=lambda x, y: ("foo", x, "bar", y),
            unbox=lambda obj: obj[3],
        )

        # Test with dict return value
        test(
            box=lambda x, y: {"foo": "bar", "a": x, "b": y},
            unbox=lambda obj: obj["b"],
        )

        # Test with list with dict return value
        test(
            box=lambda x, y: ["foo", "bar", {"a": x, "b": y}],
            unbox=lambda obj: obj[2]["b"],
        )

        # Test with dict with list return value
        test(
            box=lambda x, y: {"foo": "bar", "list": [0, x, 1, y]},
            unbox=lambda obj: obj["list"][3],
        )

    @skip_if_not_nccl
    @skip_if_not_multigpu
    def test_find_unused_parameters_kwarg(self):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        class FindUnusedParametersModule(nn.Module):
            def __init__(self):
                super(FindUnusedParametersModule, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # Return the fc3 module so that the caller can invoke it
                # outside of the forward function. While this is bad practice,
                # we can use it to trigger a reducer error.
                return (F.softmax(x, dim=1), self.fc3)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(device_id)

        def test_find_unused_parameters(find_unused_parameters, test_default=False):
            if test_default:
                model = DistributedDataParallel(
                    FindUnusedParametersModule().float().to(device_id),
                    device_ids=[device_id],
                    process_group=process_group,
                )
            else:
                model = DistributedDataParallel(
                    FindUnusedParametersModule().float().to(device_id),
                    device_ids=[device_id],
                    process_group=process_group,
                    find_unused_parameters=find_unused_parameters,
                )

            output, fc3 = model(input)
            output = fc3(output)
            loss = criterion(output, target)
            loss.backward()

        # First test that finding unused params under these conditions is to
        # trigger an error when `backward` is called (because fc3 is an unused
        # parameter and will therefore be marked ready twice).
        try:
            test_find_unused_parameters(True)
        except Exception as ex:
            self.assertTrue(
                str(ex).startswith("Expected to mark a variable ready only once."))
        else:
            self.fail("Expected exception")

        # Then test that the default behavior can be overridden by setting
        # `find_unused_parameters=False`.
        try:
            test_find_unused_parameters(False)
        except Exception as ex:
            self.fail("Unexpected exception: %s" % ex)

        # Test find_unused_parameters defaults to False
        try:
            test_find_unused_parameters(True, test_default=True)
        except Exception as ex:
            self.fail("Unexpected exception: %s" % ex)

    @skip_if_not_nccl
    @skip_if_not_multigpu
    def test_multiple_outputs_multiple_backward(self):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        class MultipleOutputModule(nn.Module):
            def __init__(self):
                super(MultipleOutputModule, self).__init__()

                def define_module():
                    return nn.Sequential(
                        nn.Linear(2, 10, bias=False),
                        nn.ReLU(),
                        nn.Linear(10, 4, bias=False),
                        nn.ReLU(),
                    )

                self.module0 = define_module()
                self.module1 = define_module()

            def forward(self, x):
                return (
                    F.softmax(self.module0(x), dim=1),
                    F.softmax(self.module1(x), dim=1),
                )

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            MultipleOutputModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(device_id)

        # Compute loss and gradients for both outputs
        output1, output2 = model(input)
        loss1 = criterion(output1, target)
        loss1.backward()
        loss2 = criterion(output2, target)
        loss2.backward()

    @skip_if_not_nccl
    @skip_if_not_multigpu
    def test_no_used_parameters(self):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        class NoUsedParameters(nn.Module):
            def __init__(self):
                super(NoUsedParameters, self).__init__()

                # Make sure this module has some parameters, only to then decide
                # to never use them from the `forward` function.
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                return x * 0.0

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            NoUsedParameters().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            find_unused_parameters=True,
        )

        batch_size = 4
        input = torch.rand([batch_size, 2], dtype=torch.float)

        # After initialization, no parameter has their gradient set.
        for p in model.parameters():
            self.assertTrue(p.requires_grad)
            self.assertIsNone(p.grad)

        # Run `forward` function.
        model(input)

        # Because none of the parameters were used, we expect reduction for
        # all parameters will be executed right when initializing the reducer.
        # Once `forward` returns, all the parameter's gradients must be set.
        for p in model.parameters():
            self.assertTrue(p.requires_grad)
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.is_tensor(p.grad))
            self.assertEqual(p.size(), p.grad.size())

    @skip_if_not_nccl
    @skip_if_not_multigpu
    def test_no_grad(self):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        class NoGradModule(nn.Module):
            def __init__(self):
                super(NoGradModule, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            NoGradModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )

        batch_size = 4
        input = torch.rand([batch_size, 2], dtype=torch.float)

        def check_no_grads():
            for p in model.parameters():
                self.assertTrue(p.requires_grad)
                self.assertIsNone(p.grad)

        # After initialization, no parameter has their gradient set.
        check_no_grads()

        # Run `forward` function with torch.no_grad()
        with torch.no_grad():
            output = model(input)
            self.assertTrue(torch.is_tensor(output))

        # No parameter should have their gradient set.
        check_no_grads()

    @skip_if_not_nccl
    @skip_if_not_multigpu
    def test_ignored_output(self):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        class IgnoredOutput(nn.Module):
            def __init__(self):
                super(IgnoredOutput, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            IgnoredOutput().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(device_id)

        # Run a few iterations where we ignore the output.
        for _ in range(4):
            output = model(input)
            del output

        # Run a few iterations where we use the output.
        for _ in range(4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()


class ReducerModule(nn.Module):
    def __init__(self):
        super(ReducerModule, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 4, bias=False)
        self.fc3 = nn.Linear(4, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, use_fc3=True):
        x = self.relu(self.fc1(x)).float()
        x = self.relu(self.fc2(x)).float()
        if use_fc3:
            x = self.fc3(x).float()
        return F.softmax(x, dim=1)


class ReducerTest(TestCase):
    def setUp(self):
        self.store = c10d.FileStore("/dev/null", 1)
        self.process_group = c10d.ProcessGroupGloo(self.store, 0, 1)

    def test_single_dtype_single_bucket(self):
        model = ReducerModule()
        parameters = list(model.parameters())
        buckets = [list(range(len(parameters)))]
        dist.Reducer([parameters], buckets, self.process_group)

    def _create_mixed_precision_model(self):
        model = ReducerModule()
        model.float()
        model.fc1.double()
        return model

    def test_multi_dtype_single_bucket(self):
        model = self._create_mixed_precision_model()

        # Raise if there are multiple types per bucket.
        # In this case we create one bucket for all parameters.
        with self.assertRaises(RuntimeError):
            parameters = [list(model.parameters())]
            buckets = [list(range(len(parameters[0])))]
            dist.Reducer(parameters, buckets, self.process_group)

    def test_multi_dtype_multi_bucket(self):
        model = self._create_mixed_precision_model()
        parameters = [list(model.parameters())]
        group_by_type = groupby(
            range(len(parameters[0])),
            key=lambda i: parameters[0][i].type())
        buckets = [list(indices) for _, indices in group_by_type]
        dist.Reducer(parameters, buckets, self.process_group)

    def _create_reducer_for_models(self, models):
        parameters = [list(model.parameters()) for model in models]
        group_by_type = groupby(
            range(len(parameters[0])),
            key=lambda i: parameters[0][i].type())
        buckets = [list(indices) for _, indices in group_by_type]
        return dist.Reducer(parameters, buckets, self.process_group)

    def test_forward_backward_single_replica(self):
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model])
        loss = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        output = loss(model(input), target)
        reducer.prepare_for_backward(output)
        output.backward()

    def test_forward_backward_multi_replica(self):
        batch_size = 10
        num_replicas = 2
        models = [self._create_mixed_precision_model() for _ in range(num_replicas)]
        reducer = self._create_reducer_for_models(models)
        loss = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.double).chunk(num_replicas)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        outputs = [models[i](input[i]) for i in range(num_replicas)]
        output = loss(torch.cat(outputs), target)
        reducer.prepare_for_backward(output)
        output.backward()

        # The reducer will have reduced the gradients for all model replicas.
        # Verify that they are equal across model replicas.
        for parameters in zip(*[model.parameters() for model in models]):
            for parameter in parameters:
                self.assertEqual(parameters[0].grad, parameter.grad)

    def test_forward_backward_unused_parameters(self):
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model])
        loss = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        output = loss(model(input, use_fc3=False), target)

        # Check that the grad of fc3 is not set.
        self.assertEqual(None, model.fc3.weight.grad)

        # Compute and accumulate gradients.
        reducer.prepare_for_backward(output)
        output.backward()

        # The reducer will have marked the grad of fc3 as ready, because
        # it doesn't show up in the autograd graph of `output`.
        # This should result in its contents being equal to zero.
        self.assertEqual(torch.zeros(model.fc3.weight.size()), model.fc3.weight.grad)

    def test_forward_backward_optimizer(self):
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model])
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        for i in range(3):
            input = torch.rand([batch_size, 2], dtype=torch.double)
            target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

            # The `zero_grad` function calls `detach_` and `zero_` on the grad
            # tensors of model parameters. If we tried to set the grad tensors
            # to a view of the reducer's bucket tensors, this would blow up.
            optimizer.zero_grad()

            # Unused parameter only in the first iteration.
            output = loss(model(input, use_fc3=(i > 0)), target)
            reducer.prepare_for_backward(output)
            output.backward()
            optimizer.step()


class ComputeBucketAssignmentTest(TestCase):
    def test_single_limit_single_dtype(self):
        tensors = [
            torch.empty([100], dtype=torch.float),
            torch.empty([200], dtype=torch.float),
            torch.empty([100], dtype=torch.float),
            torch.empty([50], dtype=torch.float),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [400])
        self.assertEqual([[0], [1], [2], [3]], result)

    def test_single_limit_multi_dtype(self):
        tensors = [
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [400])
        self.assertEqual([[0, 2], [1, 3], [4], [5]], result)

    def test_multi_limit_single_dtype(self):
        tensors = [
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [40, 80])
        self.assertEqual([[0], [1, 2], [3]], result)

    def test_multi_limit_multi_dtype(self):
        tensors = [
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [200, 400])
        self.assertEqual([[0], [1], [2, 4], [3, 5]], result)


if __name__ == '__main__':
    assert not torch.cuda._initialized, "test_distributed must not have initialized CUDA context on main process"

    run_tests()

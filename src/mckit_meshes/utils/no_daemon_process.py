"""See:

https://stackoverflow.com/questions/17223301/python-multiprocessing-is-it-possible-to-have-a-pool-inside-of-a-pool
https://stackoverflow.com/questions/52948447/error-group-argument-must-be-none-for-now-in-multiprocessing-pool
https://github.com/nipy/nipype/pull/2754
"""

import multiprocessing.pool


class Pool(multiprocessing.pool.Pool):
    # noinspection PyPep8Naming
    def Process(self, *args, **kwargs):
        proc = super(Pool, self).Process(*args, **kwargs)

        class NonDaemonProcess(proc.__class__):
            """Monkey-patch process to ensure it is never daemonized."""

            @property
            def daemon(self):
                return False

            @daemon.setter
            def daemon(self, val):
                pass

        proc.__class__ = NonDaemonProcess
        return proc

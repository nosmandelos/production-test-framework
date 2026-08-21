# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Mixin adding container support to a workload."""

from production_test_framework.docker import (
    force_remove_container,
    label_args,
    unique_container_name,
)

__all__ = ["DockerContainerMixin"]


class DockerContainerMixin:
    """
    Adds "runs its work in a container" to a workload, without changing what kind it is.

    Mix in ahead of the workload base so its :meth:`_cleanup_after_run` takes precedence::

        class MyWorkload(DockerContainerMixin, CommandWorkload):
            container_name_prefix = "my-workload"

            def build_command(self):
                return [*self.docker_run_argv("--network", "host"), "echo", "hi"]

    A mixin rather than a base class because containerisation is a capability a workload has,
    not a category it belongs to -- ``NcclWorkload`` can run its binaries directly on the host,
    which would make "is a docker workload" untrue of some of its instances.

    What it provides:

    * a container name unique per run but still known, via :func:`unique_container_name`
    * the framework label, so an orphan can be swept up by label alone
    * ``-e`` arguments built from ``env``
    * removal of the container once the run ends

    That last one is the reason this is shared rather than repeated. ``docker run --rm`` only
    removes a container that exited on its own; a stopped or timed-out run leaves it behind,
    still holding whatever it reserved. Inheriting the cleanup means a workload gets it by
    default instead of having to remember -- and forgetting would be silent, with a leaked
    ``--gpus`` container keeping those GPUs from every later run on the host.

    ``__init__`` is cooperative: it takes the container arguments and passes everything else
    along the MRO, so a workload mixing it in does not have to wire anything up by hand.
    """

    #: Leading part of a generated container name; give each workload its own so a stray
    #: container says which one started it.
    container_name_prefix = "workload"

    def __init__(
        self,
        *,
        image_name: str | None = None,
        container_name: str | None = None,
        env: dict[str, str] | None = None,
        docker_extra_args: tuple[str, ...] = (),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._image_name = image_name
        # A fixed name collides with a container an earlier run left behind, and no name at all
        # leaves nothing to remove it by. Unique and known is what makes cleanup possible.
        self._container_name = container_name or unique_container_name(self.container_name_prefix)
        self._env = dict(env or {})
        self._docker_extra_args = tuple(docker_extra_args)

    @property
    def image_name(self) -> str | None:
        return self._image_name

    @property
    def container_name(self) -> str:
        """The container this run uses, generated per instance unless one was supplied."""
        return self._container_name

    @property
    def uses_docker(self) -> bool:
        """
        Whether this run actually starts a container.

        Overridden by workloads that can also run binaries directly on the host, so cleanup
        does not go looking for a container that was never created.
        """
        return True

    def env_args(self, flag: str = "-e") -> list[str]:
        """
        Environment arguments, ``-e KEY=VALUE`` by default.

        The flag is a parameter because the same variables sometimes have to be forwarded to
        another launcher as well -- ``mpirun`` takes ``-x``.
        """
        args: list[str] = []
        for key, value in self._env.items():
            args.extend([flag, f"{key}={value}"])
        return args

    def docker_run_argv(self, *flags: str) -> list[str]:
        """
        The ``docker run`` portion of a command, up to and including the image.

        *flags* are the workload's own options -- ``--gpus``, ``--network``, ``--ulimit`` and
        so on -- inserted before the name and label so those stay adjacent and easy to read in
        a logged command line. Anything passed as ``docker_extra_args`` goes last, so it can
        override what this builds.
        """
        return [
            "docker",
            "run",
            "--rm",
            "-t",
            *flags,
            "--name",
            self._container_name,
            *label_args(self._container_name),
            *self.env_args(),
            *self._docker_extra_args,
            *([self._image_name] if self._image_name else []),
        ]

    def _cleanup_after_run(self) -> None:
        """
        Remove the container, which ``--rm`` does not do for a stopped or timed-out run.

        Cancellation terminates the ``docker run`` client, and because these runs allocate a
        TTY the client does not proxy that signal to the container: it keeps running and
        ``--rm`` never fires. A no-op after a clean run, where the container is already gone.
        """
        if self.uses_docker:
            force_remove_container(self._container_name)

"""A self-signed certificate, so a phone's camera can be used.

Browsers only hand out ``getUserMedia`` on a secure origin. ``localhost``
counts as secure; ``http://192.168.1.7:8000`` does not - which is exactly the
address a phone on the same network has to use. Without HTTPS the camera
button is simply dead there, with no useful error.

So the server can make itself a certificate. It is self-signed, so the browser
will warn once and the user has to say "proceed"; that is the honest cost of
not owning a domain. Nothing here pretends to be a real certificate authority,
and the certificate is written where only the user can read it.
"""

from __future__ import annotations

import ipaddress
import socket
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from ..utils import LOGGER

CERT_DAYS = 365


def local_addresses() -> List[str]:
    """The addresses this machine is likely reachable on.

    The certificate has to name them, or the browser rejects it outright
    instead of merely warning.
    """
    found = {"127.0.0.1"}
    try:
        # Connecting a UDP socket picks the interface that reaches the network,
        # without sending anything.
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.settimeout(0.2)
        probe.connect(("10.255.255.255", 1))
        found.add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except (OSError, socket.gaierror):
        pass
    return sorted(found)


def _san(hosts: List[str]) -> str:
    entries = ["DNS:localhost"]
    for host in hosts:
        try:
            ipaddress.ip_address(host)
            entries.append(f"IP:{host}")
        except ValueError:
            entries.append(f"DNS:{host}")
    return ",".join(entries)


def ensure_certificate(folder: Path, hosts: Optional[List[str]] = None
                       ) -> Optional[Tuple[Path, Path]]:
    """Return ``(certificate, key)``, generating them if they are not there.

    Returns ``None`` when no certificate could be made, so the caller can fall
    back to plain HTTP rather than failing to start.
    """
    folder = Path(folder)
    certificate = folder / "server.crt"
    key = folder / "server.key"
    if certificate.exists() and key.exists():
        LOGGER.info("Using the existing certificate at %s", certificate)
        return certificate, key

    hosts = hosts or local_addresses()
    folder.mkdir(parents=True, exist_ok=True)

    command = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key), "-out", str(certificate),
        "-days", str(CERT_DAYS),
        "-subj", "/CN=oneshot-fd",
        "-addext", f"subjectAltName={_san(hosts)}",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=60)
    except FileNotFoundError:
        LOGGER.error("HTTPS needs the 'openssl' command, which is not installed.")
        return None
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Could not create a certificate: %s",
                     exc.stderr.decode("utf-8", "replace").strip()[:300])
        return None
    except subprocess.TimeoutExpired:
        LOGGER.error("Creating a certificate timed out.")
        return None

    key.chmod(0o600)          # a private key is nobody else's business
    LOGGER.info("Made a self-signed certificate for %s at %s",
                ", ".join(hosts), certificate)
    LOGGER.warning("It is self-signed, so the browser will warn once. That is "
                   "expected - continue past it to use the camera.")
    return certificate, key

import socket
import ssl
import threading

from scripts.bootstrap import certificates


def test_simulator_certificate_passes_default_strict_client_verification(tmp_path):
    ca, certificate, key = certificates("sim.physicalai.internal", "127.0.0.1")
    ca_path, cert_path, key_path = (
        tmp_path / name for name in ("ca.pem", "server.pem", "server.key")
    )
    ca_path.write_bytes(ca)
    cert_path.write_bytes(certificate)
    key_path.write_bytes(key)
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_path, key_path)
    client_context = ssl.create_default_context(cafile=str(ca_path))
    client_context.verify_flags |= ssl.VERIFY_X509_STRICT
    errors = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(5)

        def accept():
            try:
                connection, _ = listener.accept()
                with server_context.wrap_socket(connection, server_side=True) as secured:
                    secured.sendall(b"verified")
            except (OSError, ssl.SSLError) as exc:
                errors.append(exc)

        thread = threading.Thread(target=accept)
        thread.start()
        with socket.create_connection(listener.getsockname(), timeout=5) as connection:
            with client_context.wrap_socket(
                connection, server_hostname="sim.physicalai.internal"
            ) as secured:
                assert secured.recv(8) == b"verified"
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors

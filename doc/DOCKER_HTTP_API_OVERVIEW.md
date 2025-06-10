# Docker Engine API: A Comprehensive Overview with `curl` Examples

## 1. Introduction to the Docker Engine API

The **Docker Engine API** is a RESTful API served by the Docker daemon (the `dockerd` process). It allows you to interact with and control the Docker daemon programmatically. Essentially, anything you can do with the `docker` command-line interface (CLI) can also be achieved by making HTTP requests to this API.

This API is crucial for:

- **Automation**: Scripting Docker operations (building images, running containers, managing networks and volumes).
- **Integration**: Integrating Docker management into other applications, CI/CD pipelines, orchestration tools (though higher-level tools like Kubernetes often abstract this), or custom dashboards.
- **Developing Docker Tools**: Building custom tools or interfaces that interact with Docker.

The API listens on a Unix socket by default (`/var/run/docker.sock` on Linux) for local connections, or it can be configured to listen on a TCP port for remote access (which requires careful security considerations).

## 2. Enabling and Accessing the API

### a. Local Access (Unix Socket - Default & Recommended for Local Use)

On Linux and macOS, the Docker daemon listens on a Unix domain socket by default. `curl` can interact with Unix sockets using the `--unix-socket` option.

- **Socket Path**: `/var/run/docker.sock` (Linux) or `~/.docker/run/docker.sock` (macOS, often a symlink).
- **Permissions**: You typically need root privileges or be part of the `docker` group to access this socket directly.

### b. Remote Access (TCP Port - Use with Extreme Caution)

Exposing the Docker API over a TCP port makes it accessible over the network. **This is inherently insecure without proper authentication and encryption (TLS)**, as anyone who can reach the port can control your Docker daemon.

**How to (cautiously) enable TCP listening (for development/testing in a secure network):**

This usually involves modifying the Docker daemon configuration. The method varies by operating system and Docker installation:

- **Linux (systemd)**:
  1. Edit `/etc/docker/daemon.json` (create if it doesn't exist):
     ```json
     {
       "hosts": ["unix:///var/run/docker.sock", "tcp://0.0.0.0:2375"]
     }
     ```
     (Using `0.0.0.0` makes it listen on all network interfaces. For a specific interface, use its IP address.)
  2. Restart Docker: `sudo systemctl restart docker`
- **Docker Desktop (Windows/macOS)**:
  - Go to Docker Desktop settings -> General.
  - Check "Expose daemon on tcp://localhost:2375 without TLS". **This is for local development only and is insecure for production or untrusted networks.**

**Accessing via TCP**:
Once enabled, you can use `curl` with `http://<docker_host_ip>:<port>`.
Default unencrypted port: `2375`
Default encrypted (TLS) port: `2376`

**SECURITY WARNING**: Exposing the Docker API without TLS is a major security risk. Always secure it with TLS if remote access is necessary in a production or untrusted environment.

## 3. API Versioning

The Docker API is versioned. It's good practice to include the API version in your requests (e.g., `/v1.41/containers/json`). If you omit the version, Docker usually defaults to the latest version it supports, but this can lead to compatibility issues if the API changes.

You can find the API version supported by your Docker daemon:

```bash
# Using Docker CLI
docker version --format '{{.Server.APIVersion}}'

# Using curl with Unix socket
curl --unix-socket /var/run/docker.sock http:/v1.41/version
# (Replace v1.41 with a recent version or try http:/version for default)
```

## 4. Common API Endpoints and `curl` Examples

All examples below assume you are either:

- Running `curl` with appropriate permissions for the Unix socket.
- Have the Docker API exposed on `localhost:2375` (for TCP examples). Adjust the host and port if different.

The API returns JSON responses. Adding `| jq` (if `jq` is installed) can pretty-print the JSON output.

### a. System Information

- **Ping**: Check if the daemon is running.

  ```bash
  # Unix Socket
  curl --unix-socket /var/run/docker.sock http:/v1.41/_ping
  # TCP
  curl http://localhost:2375/v1.41/_ping
  # Expected output: OK
  ```

- **Version Information**: Get detailed version information about the Docker daemon and components.

  ```bash
  # Unix Socket
  curl --unix-socket /var/run/docker.sock http:/v1.41/version | jq
  # TCP
  curl http://localhost:2375/v1.41/version | jq
  ```

- **System-wide Information**: Get detailed information about the Docker installation.
  ```bash
  # Unix Socket
  curl --unix-socket /var/run/docker.sock http:/v1.41/info | jq
  # TCP
  curl http://localhost:2375/v1.41/info | jq
  ```

### b. Managing Images

- **List Images**:

  ```bash
  # Unix Socket
  curl --unix-socket /var/run/docker.sock http:/v1.41/images/json | jq
  # TCP
  curl http://localhost:2375/v1.41/images/json | jq

  # List images with filters (e.g., dangling images)
  curl --unix-socket /var/run/docker.sock -G --data-urlencode 'filters={"dangling":["true"]}' http:/v1.41/images/json | jq
  ```

- **Pull an Image**: (Equivalent to `docker pull <image_name>`)
  This is a POST request. The image name is passed as a query parameter `fromImage`.

  ```bash
  # Unix Socket
  curl -X POST --unix-socket /var/run/docker.sock "http:/v1.41/images/create?fromImage=alpine&tag=latest"
  # TCP
  curl -X POST "http://localhost:2375/v1.41/images/create?fromImage=alpine&tag=latest"
  # This streams progress; add -v for more details.
  ```

- **Inspect an Image**: Get detailed information about an image.

  ```bash
  IMAGE_NAME_OR_ID="alpine" # or an image ID
  # Unix Socket
  curl --unix-socket /var/run/docker.sock http:/v1.41/images/${IMAGE_NAME_OR_ID}/json | jq
  # TCP
  curl http://localhost:2375/v1.41/images/${IMAGE_NAME_OR_ID}/json | jq
  ```

- **Remove an Image**: (Equivalent to `docker rmi <image_name>`)
  ```bash
  IMAGE_NAME_OR_ID_TO_DELETE="some_old_image"
  # Unix Socket
  curl -X DELETE --unix-socket /var/run/docker.sock http:/v1.41/images/${IMAGE_NAME_OR_ID_TO_DELETE}
  # TCP
  curl -X DELETE http://localhost:2375/v1.41/images/${IMAGE_NAME_OR_ID_TO_DELETE}
  # Add ?force=true to force removal
  ```

### c. Managing Containers

- **List Containers**:

  ```bash
  # List all containers (running and stopped)
  # Unix Socket
  curl --unix-socket /var/run/docker.sock "http:/v1.41/containers/json?all=true" | jq
  # TCP
  curl "http://localhost:2375/v1.41/containers/json?all=true" | jq

  # List only running containers
  curl --unix-socket /var/run/docker.sock "http:/v1.41/containers/json" | jq
  ```

- **Create a Container**: (Equivalent to `docker create <image_name>`)
  This is a POST request with a JSON body specifying container configuration.

  ```bash
  # Unix Socket
  curl -X POST -H "Content-Type: application/json" --unix-socket /var/run/docker.sock \
    -d '{"Image": "alpine", "Cmd": ["echo", "hello world from API"]}' \
    http:/v1.41/containers/create?name=myAlpineTest | jq
  # TCP
  curl -X POST -H "Content-Type: application/json" http://localhost:2375/v1.41/containers/create?name=myAlpineTest \
    -d '{"Image": "alpine", "Cmd": ["echo", "hello world from API"]}' | jq
  # This returns the ID of the created container.
  ```

  The JSON payload for creating containers can be very extensive, mirroring all options available in `docker run` (port mappings, volumes, environment variables, etc.). Refer to the API documentation for details.

- **Start a Container**:

  ```bash
  CONTAINER_ID_OR_NAME="myAlpineTest" # or the ID from the create step
  # Unix Socket
  curl -X POST --unix-socket /var/run/docker.sock http:/v1.41/containers/${CONTAINER_ID_OR_NAME}/start
  # TCP
  curl -X POST http://localhost:2375/v1.41/containers/${CONTAINER_ID_OR_NAME}/start
  # Returns 204 No Content on success.
  ```

- **Inspect a Container**:

  ```bash
  CONTAINER_ID_OR_NAME="myAlpineTest"
  # Unix Socket
  curl --unix-socket /var/run/docker.sock http:/v1.41/containers/${CONTAINER_ID_OR_NAME}/json | jq
  # TCP
  curl http://localhost:2375/v1.41/containers/${CONTAINER_ID_OR_NAME}/json | jq
  ```

- **Get Container Logs**:

  ```bash
  CONTAINER_ID_OR_NAME="myAlpineTest"
  # Unix Socket (add ?stdout=true&stderr=true&timestamps=true for more options)
  curl --unix-socket /var/run/docker.sock "http:/v1.41/containers/${CONTAINER_ID_OR_NAME}/logs?stdout=true&stderr=true"
  # TCP
  curl "http://localhost:2375/v1.41/containers/${CONTAINER_ID_OR_NAME}/logs?stdout=true&stderr=true"
  ```

- **Stop a Container**:

  ```bash
  CONTAINER_ID_OR_NAME="some_running_container"
  # Unix Socket
  curl -X POST --unix-socket /var/run/docker.sock http:/v1.41/containers/${CONTAINER_ID_OR_NAME}/stop
  # TCP
  curl -X POST http://localhost:2375/v1.41/containers/${CONTAINER_ID_OR_NAME}/stop
  # Add ?t=5 to specify a timeout in seconds.
  ```

- **Remove a Container**: (Equivalent to `docker rm <container_name>`)
  The container must be stopped first.
  ```bash
  CONTAINER_ID_OR_NAME_TO_DELETE="myAlpineTest"
  # Unix Socket
  curl -X DELETE --unix-socket /var/run/docker.sock http:/v1.41/containers/${CONTAINER_ID_OR_NAME_TO_DELETE}
  # TCP
  curl -X DELETE http://localhost:2375/v1.41/containers/${CONTAINER_ID_OR_NAME_TO_DELETE}
  # Add ?v=true to remove associated anonymous volumes. Add ?force=true to force remove a running container.
  ```

### d. Managing Networks

- **List Networks**:

  ```bash
  # Unix Socket
  curl --unix-socket /var/run/docker.sock http:/v1.41/networks | jq
  # TCP
  curl http://localhost:2375/v1.41/networks | jq
  ```

- **Create a Network**:

  ```bash
  # Unix Socket
  curl -X POST -H "Content-Type: application/json" --unix-socket /var/run/docker.sock \
    -d '{"Name": "my-custom-network", "Driver": "bridge"}' \
    http:/v1.41/networks/create | jq
  # TCP
  curl -X POST -H "Content-Type: application/json" http://localhost:2375/v1.41/networks/create \
    -d '{"Name": "my-custom-network", "Driver": "bridge"}' | jq
  ```

- **Inspect a Network**:

  ```bash
  NETWORK_ID_OR_NAME="my-custom-network"
  # Unix Socket
  curl --unix-socket /var/run/docker.sock http:/v1.41/networks/${NETWORK_ID_OR_NAME} | jq
  # TCP
  curl http://localhost:2375/v1.41/networks/${NETWORK_ID_OR_NAME} | jq
  ```

- **Remove a Network**:
  ```bash
  NETWORK_ID_OR_NAME_TO_DELETE="my-custom-network"
  # Unix Socket
  curl -X DELETE --unix-socket /var/run/docker.sock http:/v1.41/networks/${NETWORK_ID_OR_NAME_TO_DELETE}
  # TCP
  curl -X DELETE http://localhost:2375/v1.41/networks/${NETWORK_ID_OR_NAME_TO_DELETE}
  ```

### e. Managing Volumes

- **List Volumes**:

  ```bash
  # Unix Socket
  curl --unix-socket /var/run/docker.sock http:/v1.41/volumes | jq
  # TCP
  curl http://localhost:2375/v1.41/volumes | jq
  ```

- **Create a Volume**:

  ```bash
  # Unix Socket
  curl -X POST -H "Content-Type: application/json" --unix-socket /var/run/docker.sock \
    -d '{"Name": "my-app-data"}' \
    http:/v1.41/volumes/create | jq
  # TCP
  curl -X POST -H "Content-Type: application/json" http://localhost:2375/v1.41/volumes/create \
    -d '{"Name": "my-app-data"}' | jq
  ```

- **Remove a Volume**:
  ```bash
  VOLUME_NAME_TO_DELETE="my-app-data"
  # Unix Socket
  curl -X DELETE --unix-socket /var/run/docker.sock http:/v1.41/volumes/${VOLUME_NAME_TO_DELETE}
  # TCP
  curl -X DELETE http://localhost:2375/v1.41/volumes/${VOLUME_NAME_TO_DELETE}
  ```

## 5. Security Considerations

- **Unix Socket Permissions**: By default, the Unix socket `/var/run/docker.sock` is owned by `root` and the `docker` group. Users needing to interact with Docker without `sudo` should be added to the `docker` group. However, note that members of the `docker` group effectively have root-equivalent privileges on the host system because they can run containers with arbitrary privileges.
- **TCP Socket Security (TLS)**:
  - **NEVER expose the Docker API over an unencrypted TCP socket to an untrusted network.**
  - If remote access is required, **always** secure it using TLS (Transport Layer Security). This involves setting up a Certificate Authority (CA), server certificates/keys for the Docker daemon, and client certificates/keys for API clients.
  - Docker documentation provides detailed instructions on securing the daemon with TLS.
- **API Versioning**: Use versioned API endpoints (e.g., `/v1.41/...`) to ensure your scripts/applications don't break when the API is updated.
- **Input Validation**: If building applications that take user input to interact with the Docker API, rigorously validate all inputs to prevent injection attacks or unintended operations.
- **Principle of Least Privilege**: If an application only needs to read Docker information (e.g., list containers), ensure it uses credentials or access methods that are read-only if possible.

## 6. Further Resources

- **Official Docker Engine API Documentation**: This is the definitive source for all endpoints, parameters, request/response formats, and status codes. Search for "Docker Engine API reference".
  (e.g., [https://docs.docker.com/engine/api/v1.41/](https://docs.docker.com/engine/api/v1.41/) - replace `v1.41` with the current stable version).
- **Docker SDKs for various languages**: While `curl` is great for testing and simple scripts, for more complex applications, consider using official or community-supported Docker SDKs (e.g., `docker-py` for Python, `docker-java` for Java). These provide higher-level abstractions over the raw HTTP API.

## 7. Conclusion

The Docker Engine API provides powerful programmatic control over the Docker daemon, enabling automation, integration, and the development of custom Docker tooling. While `curl` is a useful tool for direct interaction and learning, for robust application development, using a language-specific Docker SDK is generally recommended. Always prioritize security, especially when considering remote access to the API. Understanding the API structure and its capabilities opens up a wide range of possibilities for managing and orchestrating containerized applications.

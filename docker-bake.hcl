group "default" {
  targets = ["api", "ml", "jupyter", "streamlit"]
}

target "api" {
  dockerfile = "docker/Dockerfile.api"
  tags = ["lumina-api:latest"]
  context = "."
  cache-from = ["type=registry,ref=lumina-api:cache"]
  cache-to = ["type=inline"]
  platforms = ["linux/amd64"]
}

target "ml" {
  dockerfile = "docker/Dockerfile.ml"
  tags = ["lumina-ml:latest"]
  context = "."
  cache-from = ["type=registry,ref=lumina-ml:cache"]
  cache-to = ["type=inline"]
  platforms = ["linux/amd64"]
}

target "jupyter" {
  dockerfile = "docker/Dockerfile.jupyter"
  tags = ["lumina-jupyter:latest"]
  context = "."
  cache-from = ["type=registry,ref=lumina-jupyter:cache"]
  cache-to = ["type=inline"]
  platforms = ["linux/amd64"]
}

target "streamlit" {
  dockerfile = "docker/Dockerfile.streamlit"
  tags = ["lumina-streamlit:latest"]
  context = "./frontend/streamlit-app"
  cache-from = ["type=registry,ref=lumina-streamlit:cache"]
  cache-to = ["type=inline"]
  platforms = ["linux/amd64"]
}
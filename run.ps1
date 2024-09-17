# Define the TensorFlow image for Windows CPU
$IMAGE_NAME = "nvcr.io/nvidia/tensorflow:23.12-tf2-py3"

# Check if Docker is installed
if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is not installed. Please install Docker and try again." -ForegroundColor Red
    exit 1
}

# Function to build the Docker image
function Build-DockerImage {
    param (
        [string]$ImageName
    )

    Write-Host "Building Docker image using $ImageName..."
    # Check if Docker supports buildx, otherwise fallback to standard build
    if (docker buildx version -ErrorAction SilentlyContinue) {
        docker buildx build --build-arg IMAGE_NAME=$ImageName -t mwawrzkow_midi_gan_rnn --load .
    } else {
        docker build --build-arg IMAGE_NAME=$ImageName -t mwawrzkow_midi_gan_rnn .
    }

    # Check if the build succeeded
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker build failed." -ForegroundColor Red
        exit 1
    }
}

# Function to run the Docker container
function Run-DockerContainer {
    Write-Host "Running Docker container..."
    docker run -it --rm `
        --shm-size=2g `
        --ulimit memlock=-1 `
        --ulimit stack=67108864 `
        -v "${PWD}/dockerx:/dockerx" `
        -e DISPLAY=$env:DISPLAY `
        -p 5000:5000 `
        mwawrzkow_midi_gan_rnn

    # Check if the container run succeeded
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker run failed." -ForegroundColor Red
        exit 1
    }
}

# Execute the build and run functions
Build-DockerImage -ImageName $IMAGE_NAME
Run-DockerContainer

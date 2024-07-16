# Specify the base image
FROM nvcr.io/nvidia/pytorch:19.08-py3

# Set the working directory
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install ALL requirements
RUN pip install --no-cache-dir --user -r requirements.txt

RUN pip install --user llvmlite==0.36.0

# Expose needed ports
EXPOSE 8888
EXPOSE 6006
EXPOSE 8080

# Use the official Python runtime as the base image
FROM public.ecr.aws/lambda/python:3.12-arm64


# Set the working directory
WORKDIR ${LAMBDA_TASK_ROOT}

COPY . .

# Copy the compare.py file to the zss package directory
COPY compare.py /var/lang/lib/python3.12/site-packages/zss/


# Install any necessary dependencies
RUN python3 -m pip install -r requirements.txt

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "match.lambda_handler" ]
mkdir -p python/lib/python3.12/site-        
docker run -v "$PWD":/var/task "public.ecr.aws/sam/build-python3.12" /bin/sh -c \
  "pip install 'onnxruntime>=1.19.0' tokenizers -t python/lib/python3.12/site-packages/; exit"
# How to build Tiberius
1. Git clone this repo
2. cd to this dir
3. Run `bash pre_build_script.sh`, which will perform all the time consuming download steps on your local machine, rather than wasting time doing them during the container build. These dirs will instead get copied into the container
4. Build the Dockerfile with your preferred method

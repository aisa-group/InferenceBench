
COPY inference_eval /opt/inference_eval
RUN chmod -R a+rX,go-w /opt/inference_eval && chmod a+x /opt/inference_eval/bin/launch_supervised_server.sh
COPY collect.sh capture_runtime_env.py /usr/local/bin/
COPY home /home/agent
RUN chmod +x /home/agent/task/*.sh && chown -R agent:agent /home/agent
WORKDIR /home/agent/task

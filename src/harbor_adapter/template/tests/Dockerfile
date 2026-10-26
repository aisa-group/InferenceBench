
COPY ib /opt/ib
RUN mkdir -p /opt/inference_eval && cp -a /opt/ib/src/eval/inference/bin /opt/inference_eval/bin && \
    chmod -R a+rX,go-w /opt/inference_eval && chmod a+x /opt/inference_eval/bin/launch_supervised_server.sh && \
    chown -R root:root /opt/ib && chmod 700 /opt/ib
COPY test.sh valid_eval_metrics.py snapshot.py /tests/
RUN chmod +x /tests/test.sh
WORKDIR /opt/ib

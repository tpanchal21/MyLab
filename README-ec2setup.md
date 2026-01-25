
# Set the amazon linux instance


sudo chown ec2-user:ec2-user

sudo su

mkdir src

sudo dnf install python3.13 python3.13-pip -y

sudo dnf install python3.13 python3.13-pip -y

pip3.13 install -r requirements.txt

pip3.13 install flask gunicorn openai

    Internet (HTTPS :443)
        ↓
    Gunicorn (SSL enabled)
        ↓
     Flask app

Self-signed (fast):
    openssl req -x509 -nodes -days 365 \
    -newkey rsa:2048 \
    -keyout key.pem \
    -out cert.pem

Using self-signed cert:
    nohup gunicorn \
    --workers 3 \
    --bind 0.0.0.0:443 \
    --certfile=cert.pem \
    --keyfile=key.pem \
    app:app \
    > gunicorn.log 2>&1 &

Open firewall / security group
    Allow:
    TCP 443
    (AWS: Security Group → Inbound rule)

Manage the process
    Check running
    ps aux | grep gunicorn

    Stop it
    pkill gunicorn

    Logs
    tail -f gunicorn.log
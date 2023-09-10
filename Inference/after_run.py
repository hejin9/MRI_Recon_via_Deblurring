import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os


print('After running')
# TODO Send email to the submitter

with open('/submitter.json', 'r') as f:
    submitter = json.load(f)

team_name = submitter['team_name']
email = submitter['email']

email = submitter['email']
print(f'Team Name: {team_name}\nEmail: {email}')


# 邮件发送方和接收方
sender = 'hyikesong@163.com'
receiver = email

smtp_password = os.environ.get('SMTP_PASSWORD')
if not smtp_password:
    print('No email smtp password found, skip email sending')
    exit(0)


# 创建MIMEMultipart对象，包含邮件正文和附件
msg = MIMEMultipart()
msg['From'] = sender
msg['To'] = receiver
msg['Subject'] = 'CMRxRecon Test phase passed notification'

# 邮件正文
body = 'Congratulations, your submitted Docker image has passed the test.'
msg.attach(MIMEText(body, 'plain'))


# 连接SMTP服务器
smtp_server = 'smtp.163.com'
smtp_port = 25
smtp_username = sender
server = smtplib.SMTP(smtp_server, smtp_port)
server.starttls()
server.login(smtp_username, smtp_password)

# 发送邮件
text = msg.as_string()
server.sendmail(sender, receiver, text)
server.quit()

print('Email sent successfully.')

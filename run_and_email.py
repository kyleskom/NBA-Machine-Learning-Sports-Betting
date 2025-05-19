import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Run the shell script
from email.mime.text import MIMEText

import subprocess
import os

current_dir = os.getcwd()  # Get current directory
parent_dir = os.path.dirname(current_dir)  # Get parent directory
# Define the paths
bat_path = rf"{parent_dir}/NBA-Machine-Learning-Sports-Betting/job.bat"
output_path = rf"{parent_dir}/NBA-Machine-Learning-Sports-Betting/output.txt"

def main():
    run()
    email()

def run():
    subprocess.run(
        ["cmd.exe", "/c", bat_path])

def email():
    # Regular expression to match ANSI escape codes
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    print(output_path)
    cleaned_content = ""
    # Read the file content
    try:
        with open(output_path, "r", encoding="utf-8") as file:
            file_content = file.read()

        # Clean the file content using the regex
        cleaned_content = ansi_escape.sub('', file_content)

        # Write the cleaned content back to the file
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(cleaned_content)
    except:
        with open(output_path, "r", encoding="utf-8") as file:
            file_content = file.read()

        # Clean the file content using the regex
        cleaned_content = ansi_escape.sub('', file_content)

        # Write the cleaned content back to the file
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(cleaned_content)

    # Email settings
    from_email = "blockbits30@gmail.com"
    to_email = "blockbits30@gmail.com"
    subject = "Python script output - BETTING"
    body = "Please find the attached output.txt file."
    smtp_server = "smtp.gmail.com"
    smtp_port = 587  # or 465, depending on your server
    smtp_user = "blockbits30@gmail.com"
    smtp_password = os.environ.get("emailpwd","")

    # Compose email
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Attach the output file
    filename = rf"{parent_dir}/NBA-Machine-Learning-Sports-Betting/output.txt"
    attachment = open(filename, "rb")
    part = MIMEBase("application", "octet-stream")
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename= {filename}")
    msg.attach(part)

    # Send email
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(smtp_user, smtp_password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()

    print("Email sent successfully.")

if __name__ == "__main__":
    main()
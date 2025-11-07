import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List

from opentelemetry.trace import Status, StatusCode, SpanKind

from internal import interface

class EmailClient(interface.IEmailClient):
    def __init__(
            self,
            tel: interface.ITelemetry,
            smtp_host: str,
            smtp_port: int,
            smtp_user: str,
            smtp_password: str,
            use_tls: bool = True
    ):
        self.tracer = tel.tracer()
        self.logger = tel.logger()
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.use_tls = use_tls

    async def send_email(
            self,
            to_email: str,
            subject: str,
            body: str,
            is_html: bool = True,
            attachments: Optional[List[tuple]] = None
    ) -> bool:
        with self.tracer.start_as_current_span(
                "EmailClient.send_email",
                kind=SpanKind.CLIENT,
                attributes={
                    "to_email": to_email,
                    "subject": subject,
                    "is_html": is_html
                }
        ) as span:
            try:
                result = self.__send_email_sync(
                    to_email,
                    subject,
                    body,
                    is_html,
                    attachments
                )

                if result:
                    span.set_status(Status(StatusCode.OK))
                else:
                    span.set_status(Status(StatusCode.ERROR, "Failed to send email"))

                return result

            except Exception as err:
                span.record_exception(err)
                span.set_status(Status(StatusCode.ERROR, str(err)))
                raise err

    def __send_email_sync(
            self,
            to_email: str,
            subject: str,
            body: str,
            is_html: bool,
            attachments: Optional[List[tuple]]
    ) -> bool:
        try:
            message = MIMEMultipart()
            message["From"] = self.smtp_user
            message["To"] = to_email
            message["Subject"] = subject

            if is_html:
                message.attach(MIMEText(body, "html"))
            else:
                message.attach(MIMEText(body, "plain"))

            if attachments:
                for filename, content in attachments:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(content)
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {filename}'
                    )
                    message.attach(part)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(message)

            return True

        except Exception as e:
            return False
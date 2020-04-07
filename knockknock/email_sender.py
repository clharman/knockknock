import os
import datetime
import traceback
import functools
import socket
import yagmail

DATE_FORMAT = "%m-%d %H:%M:%S"

def email_sender(recipient_emails: list, sender_email: str = None):
    """
    Email sender wrapper: execute func, send an email with the end status
    (sucessfully finished or crashed) at the end. Also send an email before
    executing func.

    `recipient_emails`: list[str]
        A list of email addresses to notify.
    `sender_email`: str (default=None)
        The email adress to send the messages. If None, use the same
        address as the first recipient email in `recipient_emails`
        if length of `recipient_emails` is more than 0.
    """
    if sender_email is None and len(recipient_emails) > 0:
        sender_email = recipient_emails[0]
    yag_sender = yagmail.SMTP(sender_email)

    def decorator_sender(func):
        @functools.wraps(func)
        def wrapper_sender(*args, **kwargs):

            start_time = datetime.datetime.now()
            host_name = socket.gethostname()
            func_name = func.__name__

            # Handling distributed training edge case.
            # In PyTorch, the launch of `torch.distributed.launch` sets up a RANK environment variable for each process.
            # This can be used to detect the master process.
            # See https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py#L211
            # Except for errors, only the master process will send notifications.
            if 'RANK' in os.environ:
                master_process = (int(os.environ['RANK']) == 0)
                host_name += ' - RANK: %s' % os.environ['RANK']
            else:
                master_process = True

            if master_process:
                contents = ['%s' % host_name,
                            '%s' % func_name,
                            '%s' % start_time.strftime(DATE_FORMAT)]
                for i in range(len(recipient_emails)):
                    current_recipient = recipient_emails[i]
                    yag_sender.send(current_recipient, 'STARTED', contents)
            try:
                value = func(*args, **kwargs)

                if master_process:
                    end_time = datetime.datetime.now()
                    elapsed_time = end_time - start_time
                    contents = ['%s' % host_name,
                                '%s' % func_name,
                                '%s' % start_time.strftime(DATE_FORMAT),
                                '%s' % end_time.strftime(DATE_FORMAT),
                                '%s' % str(elapsed_time)]

                    try:
                        str_value = str(value)
                        contents.append('%s'% str_value)
                    except:
                        contents.append('%s'% "ERROR str(value)")

                    for i in range(len(recipient_emails)):
                        current_recipient = recipient_emails[i]
                        yag_sender.send(current_recipient, 'DONE', contents)

                return value

            except Exception as ex:
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                contents = ['%s' % host_name,
                            '%s' % func_name,
                            '%s' % start_time.strftime(DATE_FORMAT),
                            '%s' % end_time.strftime(DATE_FORMAT),
                            '%s' % str(elapsed_time),
                            '%s' % ex]
                for i in range(len(recipient_emails)):
                    current_recipient = recipient_emails[i]
                    yag_sender.send(current_recipient, 'CRASHED', contents)
                raise ex

        return wrapper_sender

    return decorator_sender

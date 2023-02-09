import os
from twilio.rest import Client


class PDBExceptionHook:

    def __call__(self, exec_type, exec_val, tb):
        while tb:
            filename = tb.tb_frame.f_code.co_filename
            name = tb.tb_frame.f_code.co_name
            line_no = tb.tb_lineno
            local_vars = tb.tb_frame.f_locals
            tb = tb.tb_next

        import pdb; pdb.set_trace()


class SMSExceptionHook:
    def __init__(self):
        account_sid = "AC73f0a72f9bb2725b3e201cdeadf9649c"
        auth_token = "42b572c1518ef8ea71026fa21d82e99f"
        self.client = Client(account_sid, auth_token)

    def filter_local_vars(self, local_vars: dict):
        filtered_vars = {}

        for k, v in local_vars.items():
            type_v = str(type(v))
            if "Tensor" in type_v:
                print(type_v)
                filtered_vars[k] = v

        return filtered_vars

    def __call__(self, exc_type, exc_value, tb):
        message = """\tMessage From Neural Machine Translation Experiment (Exception Found) \n"""
        while tb:
            filename = tb.tb_frame.f_code.co_filename
            name = tb.tb_frame.f_code.co_name
            line_no = tb.tb_lineno
            message += f"\t\tFile {filename} line {line_no}, in {name} \n"

            local_vars = tb.tb_frame.f_locals
            tb = tb.tb_next

        local_vars = self.filter_local_vars(local_vars)

        message += f"\t\tLocal variables in top frame: {local_vars} \n"
        message += "=" * 80

        self.client.messages.create(
            body=message,
            from_='+18566445037',
            to='+21628407062'
        )

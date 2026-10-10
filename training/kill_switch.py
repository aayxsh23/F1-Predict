"""A kill switch so this project only ever spends free credits.

Creates (idempotent):
  * IAM user  f1-chat-runner        everything this project needs, nothing else
  * policy    f1-chat-kill-switch   deny everything
  * role      f1-budget-kill-switch the role AWS Budgets uses to attach that policy
  * budget    f1-chat-kill-switch   annual, gross usage (credits excluded); at 100% of
                                    KILL_AT_USD it AUTOMATICALLY attaches the deny
                                    policy to f1-chat-runner and emails the owner

It only stops what runs as f1-chat-runner: AWS cannot restrict the root user,
so the project's scripts must use that user's access key (created by the
owner in the console, so the secret never passes through anyone else).
Running GPU machines are not stopped by the policy; they terminate themselves
on their own timer (training/ec2.py).

    python -m training.kill_switch
"""
import json

import os

import boto3

ACCOUNT = boto3.client("sts").get_caller_identity()["Account"]
USER = "f1-chat-runner"
DENY_POLICY = "f1-chat-kill-switch"
ACCESS_POLICY = "f1-chat-runner-access"
ROLE = "f1-budget-kill-switch"
BUDGET = "f1-chat-kill-switch"
KILL_AT_USD = 175.0   # $200 of credits minus $25 for AWS's 1-2 day billing delay
EMAIL = os.environ.get("BUDGET_ALERT_EMAIL", "")  # who AWS Budgets emails; kept out of the (public) repo
BUCKET = f"f1-chat-train-{ACCOUNT}"

ACCESS = {"Version": "2012-10-17", "Statement": [
    {"Sid": "Bedrock", "Effect": "Allow", "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:Converse", "bedrock:ConverseStream",
                                                      "bedrock:List*", "bedrock:Get*", "bedrock:CreateModelInvocationJob", "bedrock:StopModelInvocationJob"], "Resource": "*"},
    {"Sid": "BatchRole", "Effect": "Allow", "Action": "iam:PassRole", "Resource": f"arn:aws:iam::{ACCOUNT}:role/f1-bedrock-batch-role"},
    {"Sid": "GpuMachine", "Effect": "Allow", "Action": ["ec2:RunInstances", "ec2:TerminateInstances", "ec2:StopInstances", "ec2:CreateTags", "ec2:Describe*",
                                                         "ec2:CreateKeyPair", "ec2:DeleteKeyPair", "ec2:CreateSecurityGroup", "ec2:AuthorizeSecurityGroupIngress",
                                                         "ec2:RevokeSecurityGroupIngress"], "Resource": "*"},
    {"Sid": "TrainingBucket", "Effect": "Allow", "Action": "s3:*", "Resource": [f"arn:aws:s3:::{BUCKET}", f"arn:aws:s3:::{BUCKET}/*"]},
    {"Sid": "ReadBilling", "Effect": "Allow", "Action": ["ce:GetCostAndUsage", "freetier:GetAccountPlanState", "freetier:GetFreeTierUsage", "pricing:GetProducts",
                                                         "budgets:ViewBudget", "budgets:DescribeBudgetAction*", "servicequotas:Get*", "servicequotas:List*",
                                                         "ssm:GetParameter"], "Resource": "*"},
]}
DENY = {"Version": "2012-10-17", "Statement": [{"Sid": "KillSwitch", "Effect": "Deny", "Action": "*", "Resource": "*"}]}


def _policy(iam, name: str, doc: dict) -> str:
    arn = f"arn:aws:iam::{ACCOUNT}:policy/{name}"
    try:
        iam.get_policy(PolicyArn=arn)
    except iam.exceptions.NoSuchEntityException:
        iam.create_policy(PolicyName=name, PolicyDocument=json.dumps(doc), Description="F1 chat project")
    return arn


def main() -> None:
    if not EMAIL:
        raise SystemExit("set BUDGET_ALERT_EMAIL to the address AWS Budgets should alert")
    iam = boto3.client("iam")
    try:
        iam.create_user(UserName=USER, Tags=[{"Key": "project", "Value": "f1-chat"}])
    except iam.exceptions.EntityAlreadyExistsException:
        pass
    access_arn, deny_arn = _policy(iam, ACCESS_POLICY, ACCESS), _policy(iam, DENY_POLICY, DENY)
    iam.attach_user_policy(UserName=USER, PolicyArn=access_arn)

    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "budgets.amazonaws.com"}, "Action": "sts:AssumeRole",
                                                     "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT}}}]}
    try:
        iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust), Description="Lets AWS Budgets switch off the F1 chat project user")
    except iam.exceptions.EntityAlreadyExistsException:
        pass
    iam.put_role_policy(RoleName=ROLE, PolicyName="attach-kill-switch", PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["iam:AttachUserPolicy", "iam:DetachUserPolicy"], "Resource": f"arn:aws:iam::{ACCOUNT}:user/{USER}",
         "Condition": {"ArnEquals": {"iam:PolicyARN": deny_arn}}}]}))
    role_arn = f"arn:aws:iam::{ACCOUNT}:role/{ROLE}"

    budgets = boto3.client("budgets", region_name="us-east-1")
    try:
        budgets.create_budget(AccountId=ACCOUNT, Budget={
            "BudgetName": BUDGET, "BudgetLimit": {"Amount": str(KILL_AT_USD), "Unit": "USD"}, "TimeUnit": "ANNUALLY", "BudgetType": "COST",
            "CostTypes": {"IncludeCredit": False, "IncludeRefund": False, "IncludeTax": True, "IncludeSubscription": True, "UseBlended": False,
                          "IncludeUpfront": True, "IncludeRecurring": True, "IncludeOtherSubscription": True, "IncludeSupport": True,
                          "IncludeDiscount": True, "UseAmortized": False}},
            NotificationsWithSubscribers=[{"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 90,
                                                            "ThresholdType": "PERCENTAGE"}, "Subscribers": [{"SubscriptionType": "EMAIL", "Address": EMAIL}]}])
    except budgets.exceptions.DuplicateRecordException:
        pass
    existing = budgets.describe_budget_actions_for_budget(AccountId=ACCOUNT, BudgetName=BUDGET)["Actions"]
    if not existing:
        import time

        for attempt in range(6):  # a brand-new role takes a few seconds to become usable
            try:
                budgets.create_budget_action(
                    AccountId=ACCOUNT, BudgetName=BUDGET, NotificationType="ACTUAL",
                    ActionType="APPLY_IAM_POLICY", ActionThreshold={"ActionThresholdValue": 100.0, "ActionThresholdType": "PERCENTAGE"},
                    Definition={"IamActionDefinition": {"PolicyArn": deny_arn, "Users": [USER]}},
                    ExecutionRoleArn=role_arn, ApprovalModel="AUTOMATIC",
                    Subscribers=[{"SubscriptionType": "EMAIL", "Address": EMAIL}])
                break
            except Exception as exc:
                if attempt == 5:
                    raise
                print(f"  waiting for the new role ({type(exc).__name__}), retrying", flush=True)
                time.sleep(10)
    actions = budgets.describe_budget_actions_for_budget(AccountId=ACCOUNT, BudgetName=BUDGET)["Actions"]
    print(json.dumps({"user": USER, "access_policy": access_arn, "kill_switch_policy": deny_arn, "budget": BUDGET,
                      "kill_at_usd": KILL_AT_USD, "action": [{"type": a["ActionType"], "status": a["Status"], "approval": a["ApprovalModel"]} for a in actions]}, indent=1))


if __name__ == "__main__":
    main()

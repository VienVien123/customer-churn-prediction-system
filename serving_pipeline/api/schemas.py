from pydantic import BaseModel

class PredictRequest(BaseModel):
    Age: float
    Gender: str
    Tenure: float
    Usage_Frequency: float
    Support_Calls: float
    Payment_Delay: float
    Subscription_Type: str
    Contract_Length: str
    Total_Spend: float
    Last_Interaction: float

class ReloadRequest(BaseModel):
    force: bool = True

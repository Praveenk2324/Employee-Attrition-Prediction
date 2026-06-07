from pydantic import BaseModel, Field
from typing import Literal


class EmployeeInput(BaseModel):
    # ── Numeric fields ──────────────────────────────────────────────────────
    Age:                      int   = Field(..., ge=18, le=60,  example=35)
    DailyRate:                int   = Field(..., ge=100, le=1500, example=800)
    DistanceFromHome:         int   = Field(..., ge=1, le=29,   example=5)
    HourlyRate:               int   = Field(..., ge=30, le=100, example=65)
    MonthlyIncome:            int   = Field(..., ge=1000,       example=5000)
    MonthlyRate:              int   = Field(..., ge=2000,       example=14000)
    NumCompaniesWorked:       int   = Field(..., ge=0, le=9,    example=2)
    PercentSalaryHike:        int   = Field(..., ge=11, le=25,  example=14)
    StockOptionLevel:         int   = Field(..., ge=0, le=3,    example=1)
    TotalWorkingYears:        int   = Field(..., ge=0, le=40,   example=10)
    TrainingTimesLastYear:    int   = Field(..., ge=0, le=6,    example=3)
    YearsAtCompany:           int   = Field(..., ge=0, le=40,   example=5)
    YearsInCurrentRole:       int   = Field(..., ge=0, le=18,   example=3)
    YearsSinceLastPromotion:  int   = Field(..., ge=0, le=15,   example=1)
    YearsWithCurrManager:     int   = Field(..., ge=0, le=17,   example=3)
    Education:                int   = Field(..., ge=1, le=5,    example=3,
                                           description="1=Below College 2=College 3=Bachelor 4=Master 5=Doctor")
    JobLevel:                 int   = Field(..., ge=1, le=5,    example=2)
    RelationshipSatisfaction: int   = Field(..., ge=1, le=4,    example=3)

    # ── Ordinal fields (1–4 scale) ──────────────────────────────────────────
    JobSatisfaction:          int   = Field(..., ge=1, le=4,    example=3,
                                           description="1=Low 2=Medium 3=High 4=Very High")
    EnvironmentSatisfaction:  int   = Field(..., ge=1, le=4,    example=2)
    WorkLifeBalance:          int   = Field(..., ge=1, le=4,    example=3)
    JobInvolvement:           int   = Field(..., ge=1, le=4,    example=3)
    PerformanceRating:        int   = Field(..., ge=3, le=4,    example=3)

    # ── Binary fields ───────────────────────────────────────────────────────
    OverTime: Literal["Yes", "No"]  = Field(..., example="No")
    Gender:   Literal["Male", "Female"] = Field(..., example="Male")

    # ── One-hot fields ──────────────────────────────────────────────────────
    Department: Literal[
        "Sales",
        "Research & Development",
        "Human Resources",
    ] = Field(..., example="Sales")

    JobRole: Literal[
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources",
    ] = Field(..., example="Sales Executive")

    MaritalStatus: Literal[
        "Single", "Married", "Divorced"
    ] = Field(..., example="Single")

    EducationField: Literal[
        "Life Sciences",
        "Medical",
        "Marketing",
        "Technical Degree",
        "Human Resources",
        "Other",
    ] = Field(..., example="Life Sciences")

    BusinessTravel: Literal[
        "Non-Travel",
        "Travel_Rarely",
        "Travel_Frequently",
    ] = Field(..., example="Travel_Rarely")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 35,
                "Department": "Sales",
                "JobRole": "Sales Executive",
                "JobSatisfaction": 2,
                "EnvironmentSatisfaction": 2,
                "WorkLifeBalance": 3,
                "JobInvolvement": 3,
                "PerformanceRating": 3,
                "OverTime": "Yes",
                "Gender": "Male",
                "MaritalStatus": "Single",
                "EducationField": "Life Sciences",
                "BusinessTravel": "Travel_Frequently",
                "Education": 3,
                "JobLevel": 2,
                "MonthlyIncome": 4500,
                "DailyRate": 800,
                "HourlyRate": 65,
                "MonthlyRate": 14000,
                "DistanceFromHome": 10,
                "NumCompaniesWorked": 3,
                "PercentSalaryHike": 13,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 8,
                "TrainingTimesLastYear": 2,
                "YearsAtCompany": 3,
                "YearsInCurrentRole": 2,
                "YearsSinceLastPromotion": 1,
                "YearsWithCurrManager": 2,
                "RelationshipSatisfaction": 3,
            }
        }
# Credit Risk Model Project

## Credit Scoring Business Understanding

1. Basel II Accord and Model Interpretability

The Basel II Capital Accord emphasizes accurate measurement and management of credit risk to ensure that banks hold sufficient capital against potential losses. In this context, models must be transparent, interpretable, and well-documented so that credit decisions can be explained to regulators, auditors, and internal risk committees. This ensures compliance, helps justify lending decisions, and allows risk management teams to understand model behavior under different scenarios.

2. Necessity of a Proxy Default Variable

Since the dataset lacks a direct “default” label, it is necessary to create a proxy variable that reflects potential credit risk based on observable customer behavior, such as Recency, Frequency, and Monetary (RFM) patterns.
Potential business risks of using a proxy include:

Misclassification of customers due to imperfect correlation with true default behavior.

Bias in credit decisions, which may lead to over-lending to risky customers or under-lending to good customers.

Regulatory scrutiny if the proxy does not reasonably approximate actual risk.

3. Trade-offs: Simple vs. Complex Models

In regulated financial contexts, there is a balance between interpretability and predictive performance:

| Aspect                   | Simple Model (e.g., Logistic Regression with WoE) | Complex Model (e.g., Gradient Boosting)              |
| ------------------------ | ------------------------------------------------- | ---------------------------------------------------- |
| Interpretability         | High – easy to explain to regulators              | Low – often a “black box”                            |
| Predictive Accuracy      | Moderate – may miss nonlinear patterns            | High – captures complex interactions                 |
| Regulatory Compliance    | Easier to justify                                 | Harder to explain; requires additional documentation |
| Maintenance & Monitoring | Easier to maintain                                | More resource-intensive                              |

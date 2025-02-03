'''Here we made a fun that will take required inputs from the user and, give those inputs to the StandardScaler to standardise and then those standardised input to the Model to predict the
Closing Price of the Stock.'''
def predict_Amazon_Close():
    try:
        import joblib
        import numpy as np
        import pandas as pd
        model=joblib.load(r"C:\Users\ACER\Deep_learning_Stock_Prediction1")
        ss=joblib.load(r"C:\Users\ACER\StandardScaler1")
        Open=float(input("Enter the Open of stock of Amazon on the date you want to predict the close of the stock: "))
        High=float(input("Enter the High of the Stock of Amazon on the date you want to predict the close of the stock: "))
        Low=float(input("Enter the Low of stock of Amazon on the date you want to predict the close of the stock: "))
        Volume=float(input("Enter the Volumne of Amazon Company on the date you want to predict the close of the stock: "))
        Day=int(input("Enter the day of the date you want to predict the close of the stock: "))
        Month=int(input("Enter month of the date you want to predict the close of the stock: "))
        Year=int(input("Enter the year of the date you want to predict the close of the stock: "))
        data=np.array([Open, High, Low, Volume, Day, Month, Year]).reshape(1, -1)
        standard_data=ss.transform(data)



        prediction=model.predict(standard_data)


        print(f"The Model Predict {prediction} as the closing price of the stock of amazon for {Day}-{Month}-{Year}")
    except ValueError as e1:
        print("Please provide valid values(numeric) for the inputs.",e1)
    except Exception as e2:
        print("Something wents wrong in the execution of the code", e2)
    finally:
        print("Thanks for using this Amazon_Stock_Prediction Deep Learning Model")

if __name__=="__main__":
    predict_Amazon_Close()
    

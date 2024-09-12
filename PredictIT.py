import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

# Define hyperparameters as constants or default values
DEFAULT_RF_N_ESTIMATORS = 100
DEFAULT_RF_MAX_DEPTH = None
DEFAULT_RANDOM_STATE = 42

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    def display_menu():
        """
        Display a menu for selecting the target audience and return corresponding CSV file names.

        Returns:
            list: List of CSV file names based on user's selection.
        """
        print("Welcome to the Registration Prediction Tool!")
        print("Please choose the target audience to predict registrations:")

        
        while True:
            print("1. IT Managers")
            print("2. Education Managers")
            print("3. Property Managers")
            print("4. Education Property Managers")
            option = input("Enter the option number: ")
            
            try:
                option = int(option)
                if option == 1:
                    return ["D19.csv", "D21.csv"]
                elif option == 2:
                    return ["SRM22.csv", "SRM23.csv"]
                elif option == 3:
                    return ["GP21.csv", "NP21.csv"]
                elif option == 4:
                    return ["MSE21.csv"]
                else:
                    print("Invalid option. Please enter a valid option.")
            except ValueError:
                print("Invalid input. Please enter a valid option.")


    def load_data(csvfiles):
        """
        Load data from CSV files and concatenate them into a single DataFrame.

        Args:
            csvfiles (str or list): File name or list of file names to be loaded.

        Returns:
            pandas.DataFrame: Concatenated DataFrame.
        """
        
        # If csvfiles is a single file name, convert it to a list
        if not isinstance(csvfiles, list):
            csvfiles = [csvfiles]

        # Read and concatenate datasets
        dfs = [pd.read_csv(file, skiprows=1) for file in csvfiles]
        df = pd.concat(dfs, ignore_index=True)

        #return the dataframe
        return df

    def parse_date(date_str):
        """
        Parse date string into a datetime object with fallback formats.

        Args:
            date_str (str): Date string in different formats.

        Returns:
            pandas.Timestamp: Parsed datetime object.
        """
        
        try:
            return pd.to_datetime(date_str, format='%d-%m-%Y')
        except ValueError:
            return pd.to_datetime(date_str, format='%d/%m/%Y')


    def preprocess_data(df, handle_outliers=True):
        """
        Preprocess the data by parsing dates, calculating week numbers, and handling outliers.

        Args:
            df (pandas.DataFrame): Input DataFrame.
            handle_outliers=True):

        Returns:
            pandas.DataFrame: Processed DataFrame with weekly registrations.
            pandas.DataFrame: DataFrame containing outlier information.
        """
        
        # Apply the custom function to the 'Created Date' column
        df['Created Date'] = df['Created Date'].apply(parse_date)

        # Calculate the week number based on the starting date
        start_date = df['Created Date'].min()
        df['week_number'] = ((df['Created Date'] - start_date).dt.days // 7) + 1

        # Count weekly registrations if the column 'Attendee Status' is either 'Attending' or 'Booker not attending'
        df['registrations'] = df['Attendee Status'].apply(lambda x: 1 if x in ['Attending', 'Booker not attending'] else 0)

        # Group by week number and calculate the number of registrations in each group
        weekly_registrations = df.groupby('week_number')['registrations'].sum().reset_index(name='registrations')

        # Identify and handle outliers using the mean method
        mean_value = weekly_registrations['registrations'].mean()
        std_dev = weekly_registrations['registrations'].std()

        # Define a threshold to identify outliers (e.g., using z-scores)
        z_threshold = 1.5  # You can adjust this threshold based on your preference

        # Identify outliers based on z-scores
        outliers = (abs((weekly_registrations['registrations'] - mean_value) / std_dev) > z_threshold)

        # Create a separate DataFrame for outliers
        outliers_df = weekly_registrations[outliers].copy()

        # If handle_outliers is True, replace outliers with mean values
        if handle_outliers:
            weekly_registrations.loc[outliers, 'registrations'] = mean_value

        """ Display the number of registrations and outliers per week
        print("Weekly Registrations:")
        print(weekly_registrations)

        print("\nOutliers:")
        print(outliers_df)
        """
        
        return weekly_registrations, outliers_df


    def feature_engineering(weekly_registrations):
        """
        Perform feature engineering by extracting relevant features for model training.

        Args:
            weekly_registrations (pandas.DataFrame): DataFrame with weekly registration information.

        Returns:
            pandas.DataFrame: Features (X).
            pandas.Series: Target variable (y).
        """
        
        # Feature engineering: Use week number as the predictor variable
        X = weekly_registrations[['week_number']]

        # Target variable
        y = weekly_registrations['registrations']

        return X, y

    def standardize_data(X):
        """
        Standardize input features using StandardScaler.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            numpy.ndarray: Standardized features.
            sklearn.preprocessing.StandardScaler: Scaler object for inverse transformations.
        """
        
        # Standardize the features using StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler

    def train_random_forest(X_train, y_train, n_estimators=DEFAULT_RF_N_ESTIMATORS, max_depth=DEFAULT_RF_MAX_DEPTH, random_state=DEFAULT_RANDOM_STATE):
        """
        Train a random forest regression model with specified hyperparameters.

        Args:
            X_train (numpy.ndarray): Training features.
            y_train (pandas.Series): Training target variable.
            n_estimators (int): Number of trees in the forest.
            max_depth (int or None): Maximum depth of the trees.
            random_state (int): Random seed for reproducibility.

        Returns:
            sklearn.ensemble.RandomForestRegressor: Trained random forest model.
        """
        
        # Create Random Forest Regressor model
        rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        rf_model.fit(X_train, y_train)
        return rf_model

    def get_starting_date():
        """
        Prompt the user to enter the starting date and return it as a datetime object.

        Returns:
            pandas.Timestamp: User-entered starting date.
        """
        
        while True:
            try:
                start_date_str = input("Enter the starting date (in the format DD-MM-YYYY): ")
                start_date = parse_date(start_date_str)
                return start_date
            except ValueError:
                print("Invalid date format. Please enter the date in the format DD-MM-YYYY.")


    def get_prediction_weeks():
        """
        Prompt the user to enter the number of weeks for predictions and return it as an integer.

        Returns:
            int: User-entered number of weeks.
        """
        
        while True:
            try:
                num_weeks = int(input("Enter the number of weeks for predictions: "))
                if num_weeks > 0:
                    return num_weeks
                else:
                    print("Please enter a valid positive number of weeks.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")


    def make_predictions(model, X):
        """
        Make predictions using the trained model.

        Args:
            model: Trained regression model.
            X (numpy.ndarray): Features for making predictions.

        Returns:
            numpy.ndarray: Predicted values.
        """
        
        # Make predictions
        predictions = model.predict(X)
        return predictions

    def evaluate_model(model, X_test, y_test):
        """
        Evaluate the model on the test set and return mean squared error and R-squared score.

        Args:
            model: Trained regression model.
            X_test (numpy.ndarray): Test features.
            y_test (pandas.Series): Test target variable.

        Returns:
            float: Mean squared error.
            float: R-squared score.
        """
        
        # Evaluate the model on the test set
        test_predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, test_predictions)
        r2 = r2_score(y_test, test_predictions)
        return mse, r2

    def plot_predictions(X_test_df, y_test, points_to_predict, rf_predictions_dict,  with_outliers=True):
        """
        Plot predictions from Random Forest models.

        Args:
            X_test_df (pandas.DataFrame): DataFrame containing test features.
            y_test (pandas.Series): Test target variable.
            points_to_predict (pandas.DataFrame): DataFrame containing points to predict.
            rf_predictions_dict (dict): Dictionary containing predictions from random forest models.
            with_outliers (bool): Flag indicating whether outliers are included.

        Returns:
            None
            
        """

        # Plot Random Forest predictions
        plt.figure(figsize=(10, 6))

        # Plot actual data points
        # plt.scatter(X_test_df['week_number'], y_test, color='black', label='Actual')

        # Plot Random Forest predictions
        for model_name, rf_predictions in rf_predictions_dict.items():
            label = f'Random Forest Model ({model_name})'
            if with_outliers:
                label += ' with Outliers'
            else:
                label += ' without Outliers'
            plt.plot(points_to_predict['week_number'], rf_predictions, label=label)
            
        plt.title(f'Random Forest Model Predictions {"with" if with_outliers else "without"} Outliers')
        plt.xlabel('Week Number')
        plt.ylabel('Registrations')
        
        # Add legend
        plt.legend()

        plt.show()

    def get_best_model_name(model_mse_dict):
        """
        Determine the best model based on the lowest Mean Squared Error (MSE).

        Args:
            model_mse_dict (dict): Dictionary containing model names as keys and their corresponding MSE values.

        Returns:
            str: Name of the best-performing model.
        """
        best_model_name = min(model_mse_dict, key=model_mse_dict.get)
        return best_model_name


    def evaluate_all_models(X_test, y_test, models):
        """
        Evaluate multiple models on the test set and return a dictionary of MSE values.

        Args:
            X_test (numpy.ndarray): Test features.
            y_test (pandas.Series): Test target variable.
            models (dict): Dictionary containing model names as keys and their corresponding models.

        Returns:
            dict: Dictionary containing model names as keys and their corresponding MSE values.
        """
        model_mse_dict = {}
        for model_name, model in models.items():
            mse, _ = evaluate_model(model, X_test, y_test)
            model_mse_dict[model_name] = mse
        return model_mse_dict

    def display_evaluation_results(model_mse_dict, model_name):
        """
        Display evaluation results for a model.

        Args:
            model_mse_dict (dict): Dictionary containing model names as keys and their corresponding MSE values.
            model_name (str): Name of the model.

        Returns:
            None
        """
        print(f"\n{'='*40}\n{f'{model_name} Evaluation Results':^40}\n{'='*40}")
        for model, mse in model_mse_dict.items():
            print(f"{model} MSE: {mse:.4f}")
        print('='*40)

    def plot_conclusion_graph(X_test_df, y_test, points_to_predict, best_model_name, best_model_predictions, with_outliers=True):
        """
        Plot the conclusion graph for model predictions.

        Args:
            X_test_df (pandas.DataFrame): DataFrame containing test features.
            y_test (pandas.Series): Test target variable.
            points_to_predict (pandas.DataFrame): DataFrame containing points to predict.
            combined_predictions_dict (dict): Dictionary containing combined predictions from different models.

        Returns:
            None

        The function generates a conclusion graph displaying predictions from different models.
        It highlights the best-performing model based on Mean Squared Error.

        The best-performing model is determined by evaluating multiple models on the test set,
        and the name of the best model is printed in the console.

        The conclusion graph shows predictions of the best model along with predictions from other models.

        Note: The best model is determined based on the lowest Mean Squared Error (MSE) on the test set.

        """
        plt.figure(tight_layout=True)
        
        # Plot actual data points
        # plt.scatter(X_test_df['week_number'], y_test, color='black', label='Actual')

        # Plot predictions for the best model
        label = f'{best_model_name} Predictions'
        if with_outliers:
            label += ' with Outliers'
        else:
            label += ' without Outliers'
            
        # Plot predictions for the best model
        plt.plot(points_to_predict['week_number'], best_model_predictions, label=f'Predictions ({best_model_name})', linestyle='--', marker='o', color='blue')

        # Display values on the graph
        for week_num, (date, prediction) in enumerate(zip(points_to_predict['predicted_date'], best_model_predictions), start=1):
            plt.annotate(f'{int(prediction)}', (week_num, prediction), textcoords="offset points", xytext=(0,10), ha='center')


        

        plt.xlabel('Week Number')
        plt.ylabel('Registrations')
        plt.legend()
        plt.title(f'Conclusion: \n \n According to the calculations made by the Prediction Tool, \n\n The Best Model is {best_model_name} Predictions {"with" if with_outliers else "without"} Outliers')
        plt.show()



    def main():
        """
        Main function to execute the registration prediction tool.
        """
        
        # Display Menu
        csvfiles = display_menu()

        # Load data
        df = load_data(csvfiles)

        # Get user input for the starting date
        start_date = get_starting_date()
        
        # Get user input for the number of weeks
        num_weeks = get_prediction_weeks()
        
        # Define points_to_predict before using it
        points_to_predict = pd.DataFrame({'week_number': range(1, num_weeks + 1)})  # Predictions for user-specified weeks
        points_to_predict['predicted_date'] = start_date + pd.to_timedelta((points_to_predict['week_number'] - 1) * 7, 'D')
        
        # Preprocess data without handling outliers
        weekly_registrations_no_outliers, _ = preprocess_data(df, handle_outliers=False)

        # Feature engineering for data without outliers
        X_no_outliers, y_no_outliers = feature_engineering(weekly_registrations_no_outliers)

        # Standardize data for data without outliers
        X_no_outliers_scaled, scaler_no_outliers = standardize_data(X_no_outliers)
        
        # Train Random Forest models without outliers
        rf_model_no_outliers_default = train_random_forest(X_no_outliers_scaled, y_no_outliers)
        rf_model_no_outliers_shallow = train_random_forest(X_no_outliers_scaled, y_no_outliers, n_estimators=150, max_depth=5, random_state=123)
        rf_model_no_outliers_deep = train_random_forest(X_no_outliers_scaled, y_no_outliers, n_estimators=200, max_depth=10, random_state=456)

        # Create a dictionary to store models and their names for easy iteration
        rf_models_no_outliers = {
            'Default': rf_model_no_outliers_default,
            'Shallow': rf_model_no_outliers_shallow,
            'Deep': rf_model_no_outliers_deep,
        }

        # X_test and y_test for without outliers cases
        X_test_df_no_outliers = pd.DataFrame(X_no_outliers_scaled, columns=['week_number'])
        X_train_no_outliers, X_test_no_outliers, y_train_no_outliers, y_test_no_outliers = train_test_split(X_no_outliers_scaled, y_no_outliers, test_size=0.2, random_state=42)
        
        
        rf_predictions_dict_no_outliers = {}
        
        # Iterate over Random Forest models without outliers, make predictions, and evaluate
        for model_name, rf_model in rf_models_no_outliers.items():
            rf_predictions_float = make_predictions(rf_model, scaler_no_outliers.transform(points_to_predict[['week_number']]))
            rf_predictions = rf_predictions_float.astype(int)
            rf_predictions_dict_no_outliers[model_name] = rf_predictions

            # Evaluate the Random Forest model on the test set without outliers
            rf_mse, rf_r2 = evaluate_model(rf_model, X_test_no_outliers, y_test_no_outliers)


            print(f"\nRandom Forest Model Predictions without Outliers ({model_name} Hyperparameters):")
            for week_num, (date, prediction) in enumerate(zip(points_to_predict['predicted_date'], rf_predictions), start=1):
                week_start_date = date.strftime("%d-%m-%Y")
                week_end_date = (date + pd.to_timedelta(6, 'D')).strftime("%d-%m-%Y")
                print(f'Predicted Registrations for Week {week_num} ({week_start_date} - {week_end_date}): {int(prediction)}')

            # Display results in a table
            print(f"\n{'='*40}\n{'Random Forest Model Evaluation without Outliers ({model_name} Hyperparameters) on Test Set':^40}\n{'='*40}".format(model_name=model_name))

            print(f"{'Mean Squared Error':<25} {rf_mse:.4f}")
            print(f"{'R^2 Score':<25} {rf_r2:.4f}")
            print('='*40)

        # Convert X_test_no_outliers to DataFrame
        X_test_df_no_outliers = pd.DataFrame(X_test_no_outliers, columns=['week_number'])

        # Plot predictions without outliers
        plot_predictions(X_test_df_no_outliers, y_test_no_outliers, points_to_predict, rf_predictions_dict_no_outliers, with_outliers=False)

        #------------------------with OUTLIERS ------------------------
            
        # Preprocess data with handling outliers
        weekly_registrations_with_outliers, _ = preprocess_data(df, handle_outliers=True)
        
        # Feature engineering for data with outliers
        X_with_outliers, y_with_outliers = feature_engineering(weekly_registrations_with_outliers)
        
        # Standardize data for data with outliers
        X_with_outliers_scaled, scaler_with_outliers = standardize_data(X_with_outliers)

        # Train Random Forest models with outliers
        rf_model_with_outliers_default = train_random_forest(X_with_outliers_scaled, y_with_outliers)
        rf_model_with_outliers_shallow = train_random_forest(X_with_outliers_scaled, y_with_outliers, n_estimators=150, max_depth=5, random_state=123)
        rf_model_with_outliers_deep = train_random_forest(X_with_outliers_scaled, y_with_outliers, n_estimators=200, max_depth=10, random_state=456)

        # Create a dictionary to store models and their names for easy iteration
        rf_models_with_outliers = {
            'Default': rf_model_with_outliers_default,
            'Shallow': rf_model_with_outliers_shallow,
            'Deep': rf_model_with_outliers_deep,
        }
        
        # X_test and y_test for with outlier cases
        X_test_df_with_outliers = pd.DataFrame(X_with_outliers_scaled, columns=['week_number'])
        X_train_with_outliers, X_test_with_outliers, y_train_with_outliers, y_test_with_outliers = train_test_split(X_with_outliers_scaled, y_with_outliers, test_size=0.2, random_state=42)

        rf_predictions_dict_with_outliers = {}

        # Iterate over Random Forest models with outliers, make predictions, and evaluate
        for model_name, rf_model in rf_models_with_outliers.items():
            rf_predictions_float = make_predictions(rf_model, scaler_with_outliers.transform(points_to_predict[['week_number']]))
            rf_predictions = rf_predictions_float.astype(int)
            rf_predictions_dict_with_outliers[model_name] = rf_predictions

            # Evaluate the Random Forest model on the test set with outliers
            rf_mse, rf_r2 = evaluate_model(rf_model, X_test_with_outliers, y_test_with_outliers)

            print(f"\nRandom Forest Model Predictions with Outliers ({model_name} Hyperparameters):")
            for week_num, (date, prediction) in enumerate(zip(points_to_predict['predicted_date'], rf_predictions), start=1):
                week_start_date = date.strftime("%d-%m-%Y")
                week_end_date = (date + pd.to_timedelta(6, 'D')).strftime("%d-%m-%Y")
                print(f'Predicted Registrations for Week {week_num} ({week_start_date} - {week_end_date}): {int(prediction)}')

            # Display results in a table
            print(f"\n{'='*40}\n{'Random Forest Model Evaluation with Outliers ({model_name} Hyperparameters) on Test Set':^40}\n{'='*40}".format(model_name=model_name))

            print(f"{'Mean Squared Error':<25} {rf_mse:.4f}")
            print(f"{'R^2 Score':<25} {rf_r2:.4f}")
            print('='*40) 

        # Convert X_test_no_outliers to DataFrame
        X_test_df_with_outliers = pd.DataFrame(X_test_with_outliers, columns=['week_number'])
        
        # Plot predictions with outliers
        plot_predictions(X_test_df_with_outliers, y_test_with_outliers, points_to_predict, rf_predictions_dict_with_outliers, with_outliers=True)

        # ----------------------- END --------------------------

        # Evaluate models and get MSE values (without Outliers)
        best_model_mse_dict_no_outliers = evaluate_all_models(X_test_df_no_outliers, y_test_no_outliers, rf_models_no_outliers)

        # Determine the best model (without outliers)
        best_model_name, best_model_mse_dict, best_rf_predictions_dict, best_X_test_df, best_y_test = \
            (get_best_model_name(best_model_mse_dict_no_outliers),
             best_model_mse_dict_no_outliers,
             rf_predictions_dict_no_outliers,
             X_test_df_no_outliers,
             y_test_no_outliers)
        
        # Display evaluation results for the best model (without Outliers)
        display_evaluation_results(best_model_mse_dict, f"Best Model (Without Outliers)")

        # Calculate and display the sum of total predictions made
        best_model_predictions_sum = int(sum(rf_predictions_dict_no_outliers[best_model_name]))
        print(f"Total sum of predictions made by the best model ({best_model_name}): {best_model_predictions_sum}")

        # Plot the conclusion graph for the best model (without Outliers)
        plot_conclusion_graph(best_X_test_df, best_y_test, points_to_predict, best_model_name, rf_predictions_dict_no_outliers[best_model_name], False)

        # Evaluate models and get MSE values (with Outliers)
        best_model_mse_dict_with_outliers = evaluate_all_models(X_test_df_with_outliers, y_test_with_outliers, rf_models_with_outliers)
        
        # Determine the best model (with outliers)
        best_model_name, best_model_mse_dict, best_rf_predictions_dict, best_X_test_df, best_y_test = \
            (get_best_model_name(best_model_mse_dict_with_outliers),
                  best_model_mse_dict_with_outliers,
                  rf_predictions_dict_with_outliers,
                  X_test_df_with_outliers,
                  y_test_with_outliers)

        # Display evaluation results for the best model (with Outliers)
        display_evaluation_results(best_model_mse_dict, f"Best Model (With Outliers)")

        # Calculate and display the sum of total predictions made
        best_model_predictions_sum = int(sum(rf_predictions_dict_with_outliers[best_model_name]))
        print(f"Total sum of predictions made by the best model ({best_model_name}): {best_model_predictions_sum}")

        # Plot the conclusion graph for the best model (with Outliers)
        plot_conclusion_graph(best_X_test_df, best_y_test, points_to_predict, best_model_name, rf_predictions_dict_with_outliers[best_model_name])

        # Display a conclusion message
        print(f"\n{'='*40}\n{'Conclusion':^40}\n{'='*40}")
        print(f"The best-performing model based on Mean Squared Error is: {best_model_name}")
        print("\nThank you for using the registration prediction tool! If you have any questions or feedback, feel free to reach out.")

    if __name__ == "__main__":
        main()


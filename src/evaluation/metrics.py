from sklearn.metrics import classification_report


def evaluate(model, X_test, y_test, label_encoder):

    y_pred = model.predict(X_test)

    print("\n[+] Evaluation:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_
        )
    )
import numpy as np

class CustomLinearRegression:
    def __init__(self, lr=0.1, epochs=10000, lambda_=0.01, tol=1e-6):
        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_
        self.tol = tol
        self.w = None
        self.b = None
        self.loss_history = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # Momentum giúp thuật toán đi qua các vùng phẳng và tiến tới Minimum nhanh hơn
        v_w = np.zeros(n_features)
        v_b = 0.0
        beta = 0.9 # Hệ số momentum
        
        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b
            error = y_pred - y
            
            loss = np.mean(error**2) + self.lambda_ * np.sum(self.w**2)
            self.loss_history.append(loss)
            
            dw = (2/n_samples) * np.dot(X.T, error) + 2 * self.lambda_ * self.w
            db = (2/n_samples) * np.sum(error)
            
            # Cập nhật bằng Momentum SGD
            v_w = beta * v_w + (1 - beta) * dw
            v_b = beta * v_b + (1 - beta) * db
            
            self.w -= self.lr * v_w
            self.b -= self.lr * v_b
            
            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                print(f"Custom Model: Dừng sớm tại vòng lặp {epoch} do loss đã hội tụ.")
                break
                
            if epoch % 500 == 0:
                print(f"Epoch {epoch:4d}: Loss = {loss:.4f}")
                
    def predict(self, X):
        return np.dot(X, self.w) + self.b

class CustomLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000, lambda_=0.01, tol=1e-6):
        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_
        self.tol = tol
        self.w = None
        self.b = None
        self.loss_history = []
        
    def _sigmoid(self, z):
        # np.clip để tránh overflow trong hàm exp
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # Sử dụng Momentum SGD cho hội tụ nhanh
        v_w = np.zeros(n_features)
        v_b = 0.0
        beta = 0.9 
        
        for epoch in range(self.epochs):
            # Tính Linear model
            linear_model = np.dot(X, self.w) + self.b
            # Áp dụng Sigmoid
            y_predicted = self._sigmoid(linear_model)
            
            # Binary Cross Entropy Loss + L2 Regularization (Ridge)
            # Thêm epsilon để tránh log(0)
            epsilon = 1e-9
            loss = -1/n_samples * np.sum(y * np.log(y_predicted + epsilon) + (1 - y) * np.log(1 - y_predicted + epsilon)) 
            loss += self.lambda_ * np.sum(self.w**2)
            self.loss_history.append(loss)
            
            # Tính gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + 2 * self.lambda_ * self.w
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Cập nhật w, b với Momentum
            v_w = beta * v_w + (1 - beta) * dw
            v_b = beta * v_b + (1 - beta) * db
            
            self.w -= self.lr * v_w
            self.b -= self.lr * v_b
            
            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                print(f"Logistic Model: Dừng sớm tại vòng lặp {epoch} do loss đã hội tụ.")
                break
                
            if epoch % 500 == 0:
                print(f"Epoch {epoch:4d}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        linear_model = np.dot(X, self.w) + self.b
        return self._sigmoid(linear_model)
        
    def predict(self, X, threshold=0.5):
        y_predicted_cls = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in y_predicted_cls]

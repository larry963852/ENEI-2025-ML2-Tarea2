# Assignment 2: Principal Component Analysis, Neural Networks

Deadline: Tuesday, November 4th, 2025, 23:59

Environment: Python, numpy, pandas, matplotlib, scikit-learn, pytorch.

---

## Programming Exercises

### Part I: Eigenfaces for Face Recognition

1. **Load the Training and Test Sets**

   Load the training images contained in `data` into a matrix **X**.
   There are 540 training images, each with resolution $50 \times 50$. Flatten each image into a 2500-dimensional vector. Thus, **X** should have shape **$540 \times 2500$**, where each row is a flattened face image.

   Similarly, build the test matrix **$X_{test}$**, which should have shape **$100 \times 2500$**.

   Display an example training and test image in grayscale.

   Example code snippet for loading the training data:

   ```python
   import numpy as np
   from matplotlib import pylab as plt
   import matplotlib.cm as cm
   import imageio

   train_labels, train_data = [], []
   for line in open('./data/train.txt'):
       im = imageio.v2.imread("" + line.strip().split()[0])
       train_data.append(im.reshape(2500,))
       train_labels.append(line.strip().split()[1])
   train_data, train_labels = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)

   print(train_data.shape, train_labels.shape)
   plt.imshow(train_data[10, :].reshape(50,50), cmap = cm.Greys_r)
   plt.show()
   ```

2. **Average Face**

   Compute the *average face* vector $ \mu $ by averaging all rows of **$X$**.
   Display this average face as a grayscale image.

3. **Mean Subtraction**

   Subtract the average face $\mu$ from each row of **$X$**, i.e., replace each image vector $x_i$ with $x_i - \mu$.
   Display an example mean-subtracted image.
   Apply the same mean subtraction to **$X_{test}$**, using the same $\mu$.
   From now on, for training and testing, you should use the demeaned matrix.

4. **Eigenfaces**

   Compute the eigendecomposition of $X^T X = V \Lambda V^T$ to obtain eigenvectors.
   The rows of $V^T$ correspond to eigenfaces.

   Display 10 eigenfaces as grayscale images.

   Note: Eigenvectors may be complex-valued. You will need to convert them to real values before displaying (e.g., using `np.real`).

5. **Eigenface Features**

   The top $r$ eigenfaces span an $r$-dimensional **face space**.
   Represent an image vector $z$ in this space as:
   
$$
   f = [v_1, v_2, \ldots, v_r]^T z
$$

   Write a function to compute:

   * **$F$**: feature matrix for training data (shape: $540 \times r$)
   * **$F_{test}$**: feature matrix for test data (shape: $100 \times r$)

   by multiplying **$X$** and **$X_{test}$** with the top $ r$ eigenfaces.

6. **Face Recognition**

   Use **logistic regression** (e.g., from `scikit-learn`) for classification.

   * Extract features using $ r = 10 $ (supress the intercept, as it is not necessary because the matrix is demeaned)
   * Train logistic regression on **$F$** and evaluate on **$F_{test}$**
   * Report classification accuracy on the test set
   * Then repeat for $ r = 1, 2, \ldots, 200 $ and plot accuracy as a function of $ r $

7. **Low-Rank Reconstruction Loss**

   Reconstruct approximations $ X' $ from the features by multiplying:

$$
X' = F \cdot \text{(top } r \text{ eigenfaces)}
$$

   Compute and plot the average Frobenius distance:

$$
   d(X, X') = \sqrt{\text{tr}((X - X')^T (X - X'))}
$$

   for $ r = 1, 2, \ldots, 200 $.

### Part II: Neural Networks

Modify the example on Convolutional Neural Networks shown in the practical sessions, to use the original MNIST dataset. Create and train all models shown, and plot their convergence curves. To download the MNIST data, use:

```python
mnist_train = datasets.MNIST(
    "/content/sample_data", download = True, train = True,
    transform = transforms.ToTensor()
)
mnist_test = datasets.MNIST(
    "/content/sample_data", train = False, transform = transforms.ToTensor()
)
```

---

### Deliverables

  * You must fork the original repository, and turn in a link to your group's repository.
  * This fork must have a Jupyter notebook in the src folder, which contains all the code to solve each of the problems.
  * For the written commentary, you may choose between presenting it in Markdown cells within the Jupyter notebook or creating a separate README.md inside the src folder.


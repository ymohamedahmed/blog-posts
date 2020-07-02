# Understanding principal component analysis 

In this post, we'll take a deep dive into PCA, from both a mathematical and implementation perspective.
We'll derive the equation from the ground up, look at how we can compute it and finally end with what it can be used for. This post is primarily targeted at those with a basic understanding of PCA but want to know the assumptions it relies on, its properties and derive how it can be computed.

## The optimal coding perspective

### Finding a low dimensional representation
PCA can be thought of as finding a low-dimensional representation of a set of vectors. Given points in an _n_-dimensional space, we might wish to find some new _k_-dimensional space (with _k_ < _n_) which captures as much of the _essence_ of the original space as possible. The exact definition of capturing the '_essence_' is subject to design, however, we can consider it from multiple perspectives. 

![](diagrams/pca.svg)

### The notion of reconstruction error
If we take our low-dimensional representation and attempt to recover the original _n_-dimensional vector of each point, we could measure how much each point varies from its reconstruction. The difference between each reconstruction and the original, is one way of measuring the effectiveness of our new _k_-dimensional space and is the approach taken by PCA.  Naturally, this requires a definition of a _similarity_ between two matrices. If we have a matrix $X$ of our original points and our reconstruction $X'$ then we can define the difference between them as a sum of the square of errors $\sum_{i,j} (X-X')_{i,j}^2$.

This quantity is known as the _Frobenius_ norm of the matrix $||X-X'||_F$ and is essentially an extension of the L2 ($||\mathbf{v}||_2^2 =  \sum_i \mathbf{v}_i^2 = \mathbf{v}^\top \mathbf{v}$) norm from vectors to matrices. It is just a fancy name for squaring every element in a matrix and taking their sum. Crucially, however, we can see that the Frobenius norm of a matrix, $A$, is precisely equivalent to $Tr[A^\top A]$ (see the illustration below).

<!-- ![](diagrams/media/videos/frobenius_animation/1440p60/output.gif) -->
![](diagrams/media/videos/frobenius_animation/1440p60/output.gif)
<figure class="image">
<!--     <img src="diagrams/media/videos/frobenius_animation/1440p60/output_smaller.gif"> -->
    <figcaption>Showing the relationship between the Frobenius norm and trace operator: $||A||_F^2 = \text{Tr}(A^\top A)$</figcaption>
</figure>

As a result, our error of interest, can be computed as $Tr[(X-X')^\top (X-X')]$, this will come in handy since the trace operator comes with a bunch of neat tricks for manipulating the matrices involved.




### PCA Assumptions
We've defined how we're going to evaluate this reconstruction, but not at all the means of performing the coding or its inverse.

PCA chooses to implement both the encoding and decoding as a matrix multiplication. So we can think of PCA as finding a matrix $D$ that will transform our input, $X$ to our coded version $C$ by a matrix multiplication (i.e. $C = XD$).
The matrix, $X$, being our original data of $m$ rows and $n$ columns.

We might choose, however, to use some matrix, $D_k$ which reduces the number of dimensions of our data from $m$ to $k$. To make it easier, we can tag the matrix with the new number of dimensions. If this is the case, then $D_k$ is of size $n \times k$ and so that the resulting coding, $C$, is of size $m \times k$.

One of the most crucial assumptions made by PCA is that the transformation matrix, $D_k$, has __orthonormal__ columns. Criticially, this does not, technically, make it an __orthogonal__ matrix since it may not be square and the rows may not be orthonormal.

This assumption is useful since it simplifies the reconstruction process, in fact, $X'$ can be computed as $XD_kD_k^\top$, i.e. we can use the transpose of the encoding matrix to perform the decoding.

To understand exactly why the reconstruction can be performed by the transpose of the matrix let's consider a code $\mathbf{z}$ which we've generated as $\mathbf{z}=\mathbf{D}\mathbf{x}$ and are decoding using $\mathbf{D}^\top$.

The reconstruction error is $(\mathbf{x}-\mathbf{D}^\top\mathbf{z})^\top(\mathbf{x}-\mathbf{D}^\top\mathbf{z}) = \mathbf{x}^\top\mathbf{x} - \mathbf{x}^\top\mathbf{D}^\top\mathbf{z}-\mathbf{z}^\top\mathbf{D}\mathbf{x} + \mathbf{z}^\top\mathbf{D}\mathbf{D}^\top\mathbf{z}$.

We can take the derivative with respect to the code $\mathbf{z}$, $\nabla_\mathbf{z} = -2\mathbf{D}\mathbf{x} + 2\mathbf{D}\mathbf{D}^\top\mathbf{z}$, which given the definition of $\mathbf{z}$ is equal to $-2\mathbf{D}\mathbf{x} + 2\mathbf{D}\mathbf{D}^\top\mathbf{D}\mathbf{x}$.

We defined our matrix to have orthonormal columns so we know that $\mathbf{D}^\top\mathbf{D} = \mathbf{I}$ because the dot product of each column with itself will be one. Hence, our gradient is  $-2\mathbf{D}\mathbf{x} + 2\mathbf{D}\mathbf{I}\mathbf{x} = \mathbf{0}$. This tells us that our encoding/decoding system is at a stationary point in the reconstruction error. Note that this only tells us for an individual point that was encoded with a matrix that has orthonormal columns, decoding with the transpose is a good idea. It doesn't tell us anything about how to pick $\mathbf{D}$ or what happens if you use it for multiple points.

Given these assumptions, we can reframe our problem as finding the coding matrix which minimises the reconstruction error.

This means that formally, for some given value $k$, we wish to discover the matrix $D_k^*$. 
$$D_k^* = argmin_{D_k} ||X-X'||_F =  argmin_{D_k} ||X-XD_kD_k^\top||_F$$

### Discovering the coding function

Given that we wish to minimise the reconstruction error, let us attempt to discover the transformation precisely capable of this.








We'll need a few tricks to get us there: 

- $(A+B)^\top = A^\top + B^\top$
- $(AB)^\top = B^\top A^\top$
- $(A^\top)^\top = A$

From the visual illustration, recall that we can write the reconstruction error as $\text{Tr}((X-XD_kD_k^\top)^\top(X-XD_kD_k^\top))$.

By using the rules  this can be expanded into $$\text{argmin}_{D_k}\text{Tr}(X^\top X -X^\top XD_kD_k^\top - D_kD_k^\top X^\top X + D_kD_k^\top X^\top XD_kD_k^\top)$$

We are, however, only interested in the effect of the matrix $D_k$ so we'll axe the first term $$\text{argmin}_{D_k}\text{Tr}(-X^\top XD_kD_k^\top - D_kD_k^\top X^\top X + D_kD_k^\top X^\top XD_kD_k^\top)$$


Crucially, however, the trace operator has two useful properties for us: 
 - it's insensitive to cyclic permutations (i.e. $\text{Tr}(ABC) = \text{Tr}(CAB) = \text{Tr}(BCA)$)
 - the trace of a sum of matrices is the sum of their traces (i.e $\text{Tr}(\sum_i \mathbf{A}_i) = \sum_i \text{Tr}(\mathbf{A}_i)$)
 
Given this, we can write the equation of interest as: $$\text{argmin}_{D_k}-2\text{Tr}(-X^\top XD_kD_k^\top)+ \text{Tr}(D_kD_k^\top X^\top XD_kD_k^\top)$$.

Furthermore, since we define the columns of $D_k$ as being orthonormal, $\mathbf{D}_k^\top \mathbf{D}_k = \mathbf{I}$ and so we can rewrite the second term as $\text{Tr}(\mathbf{D}_k^\top \mathbf{X}^\top  \mathbf{X}\mathbf{D}_k)$ which matches the form of the first term (from permuting it) and therefore can be written as the following maximisation.
$$\text{argmax}_{\mathbf{D}_k} \text{Tr}(\mathbf{D}_k^\top \mathbf{X}^\top  \mathbf{X}\mathbf{D}_k)$$

This is sets up exactly the quantity that PCA is attempting to maximise.

#### Relationship with eigendecomposition

Having happily derived the maximisation problem, I'll do my best to convince you that this is, in fact, maximised by having the k-columns of $\mathbf{D}_k$ as the k eigenvectors of $\mathbf{X}^\top\mathbf{X}$ with largest eigenvalues. This will roughly take the form of an inductive proof. So let's start with considering $k=1$, that is if we had the choice of using a single vector, what would we go with?

##### Base case
We'll use the lowercase, $\mathbf{d}$, to emphasise that it's a vector rather than matrix. So let's try and tackle: $$\text{argmax}_{\mathbf{d}} \text{Tr}(\mathbf{d}^\top \mathbf{X}^\top  \mathbf{X}\mathbf{d})$$
The trace of a scalar, is defined as itself, so we wish to find $$\text{argmax}_{\mathbf{d}} \mathbf{d}^\top \mathbf{X}^\top  \mathbf{X}\mathbf{d}$$

However, recall that we mandated as our first assumption that the columns of $\mathbf{D}$ are orthonormal and hence we know that $\mathbf{d}^\top\mathbf{d} = 1$.

This allows us to reframe this as a constrained optimisation problem and use Lagrange multipliers to discover which vector minimises the quantity of interest whilst satisfying the norm constraint. For convenience, we'll use $\mathbf{A} = \mathbf{X}^\top  \mathbf{X}$.

Hence we can write this as:
$$L(\mathbf{d}, \mathbf{A}, \lambda) = \mathbf{d}^\top \mathbf{A}\mathbf{d} - \lambda(\mathbf{d}^\top \mathbf{d} - 1)$$


> TL;DR Lagrange multipliers are just a technique for optimising a function over a specific region, in our case we're not looking for all vectors but only ones with a norm of 1. We rewrite the function to include the constraint and take the derivative of the new function, known as the _Lagrangian_.


By taking the derivative of $L$ we can find the solution. $$\frac{\partial L}{\partial \mathbf{d}} = 2\mathbf{A}\mathbf{d} - 2\lambda\mathbf{d}$$

 The derivative is precisely zero when $\mathbf{A}\mathbf{d} = \lambda\mathbf{d}$, that is when $\mathbf{d}$ is an eigenvector of the matrix $\mathbf{A}$.
Furthermore, we can see that the quantity in fact takes on the associated eigenvalue since $\mathbf{d}^\top \mathbf{A}\mathbf{d} = \mathbf{d}^\top \lambda\mathbf{d} = \lambda $, where $\lambda$ is the associated eigenvalue. This is always possible since we can mandate that the eigenvector has a unit norm.

Hence, we can see here that for the base case, the best vector is the eigenvector with the largest eigenvalue.

##### Inductive step

Having showcased our hypothesis for $k=1$, let's see what happens for other values of $k$. Precisely, let us show that if our hypothesis is true for some arbitrary $k$ then it is also true for some $k+1$. That is the best encoding matrix with $k+1$ columns, consists of the $k+1$ unit eigenvectors with the largest associated eigenvalues.

We are now interested in the following problem: 

$$\text{argmax}_{\mathbf{D}_{k+1}} \mathrm{Tr}(\mathbf{D}_{k+1}^\top \mathbf{X}^\top  \mathbf{X}\mathbf{D}_{k+1})$$

The key insight comes from thinking of the $\mathbf{D}_{k+1}$ as the previous matrix,$\mathbf{D}_{k}$, for which we have assumed the property to hold, and a new column which we would like to show corresponds the eigenvector with the $(k+1)$-th largest eigenvalue.

![](diagrams/pca-proof.svg)


The matrix of interest is split into two new matrices, $\Lambda_k$ and $\Lambda_{k+1}$ as shown in the diagram. So we can rewrite the maximisation and expand it.

$$\text{argmax}_{\mathbf{\Lambda}_{k+1}} \mathrm{Tr}((\mathbf{\Lambda}_{k} + \mathbf{\Lambda}_{k+1})^\top \mathbf{X}^\top  \mathbf{X}(\mathbf{\Lambda}_{k} + \mathbf{\Lambda}_{k+1}))$$

$$\text{argmax}_{\mathbf{\Lambda}_{k+1}} \mathrm{Tr}[\mathbf{\Lambda}_{k}^\top \mathbf{X}^\top  \mathbf{X}\mathbf{\Lambda}_{k} + \mathbf{\Lambda}_{k+1}^\top \mathbf{X}^\top  \mathbf{X}\mathbf{\Lambda}_{k} + \mathbf{\Lambda}_{k}^\top \mathbf{X}^\top  \mathbf{X}\mathbf{\Lambda}_{k+1} + \mathbf{\Lambda}_{k+1}^\top \mathbf{X}^\top  \mathbf{X}\mathbf{\Lambda}_{k+1}]$$

We'll slowly get rid of three of these four terms and show our initial hypothesis holds.
- First term: is a constant from our perspective and so we don't have to consider it
- Second and third term: I'll try to convince you from the diagram that both of these have zero entries along the diagonal. The diagram shows why it's true for the second term and by symmetry we can see it's also true for the third term. Since they have zeroes along the diagonal they will have a trace of zero and hence by splitting the trace we end up with $\mathrm{Tr}[\mathbf{\Lambda}_{k+1}^\top \mathbf{X}^\top  \mathbf{X}\mathbf{\Lambda}_{k+1}]$ as the final maximisation.

![](diagrams/pca-proof-2.svg)

However, if we look carefully enough we can see that the fourth term has zeroes everywhere other than in the bottom right corner, which I denote by $\mu$.

<img src="diagrams/pca-proof-3.svg" align="center"/>

Furthermore, from the diagram we can see that $\mu = \mathbf{a}'\mathbf{d}_{k+1}$ and that $\mathbf{a}'=\mathbf{d}_{k+1}^\top\mathbf{A}$.

Now we can, finally, reframe our inductive case as $$\text{argmax}_{\mathbf{d}_{k+1}} \mathbf{d}_{k+1}^\top\mathbf{X}^\top\mathbf{X}\mathbf{d}_{k+1}$$

However, recall that we mandated that all the columns of the encoding matrix are orthonormal and hence, this condition is maximised, precisely, by the eigenvector with (k+1)-th largest eigenvalue. It is important to note that the eigenvectors (corresponding to different eigenvalues) are orthogonal here, since $\mathbf{X}^\top \mathbf{X}$ is real and symmetric. Given that we chose our base case as an eigenvector, we know that if we pick the $k+1$ vector as a different eigenvector, it will necessarily be orthogonal to all the other columns.

Finally, this induction cannot continue indefinitely, since we cannot have $n+1$ orthogonal $n$-dimensional vectors. This is intuitive since PCA is designed to find a lower-dimensional representation not increase the number of dimensions.



I hope I have convinced you that the we can find the principal components by eigendecomposition of $\mathbf{X}^\top \mathbf{X}$.

That is, if we write it as $\mathbf{X}^\top \mathbf{X} = \mathbf{V}\text{diag}(\mathbf{\lambda})\mathbf{V}$, with the matrix $V$ containing the eigenvectors as columns and with eigenvalues in the vector $\mathbf{\lambda}$ then we can pick our encoding matrix as the $k$ eigenvectors with largest eigenvalues.

### Relationship with SVD

Recall that singular value decomposition is defined as rewriting a matrix, $\mathbf{A}$, as a product, $\mathbf{U}\mathbf{D}\mathbf{V}^\top$, of orthogonal matrices, $\mathbf{U}$ and $\mathbf{V}$ and a diagonal matrix $\mathbf{D}$.

Crucially, however, $\mathbf{V}$ is defined as the eigenvectors of the matrix $\mathbf{A}^\top\mathbf{A}$. Sound familiar? That's precisely the eigenvectors that we need for the PCA encoding.

Hence, we know that if $\mathbf{X}^\top \mathbf{X} = \mathbf{V}\text{diag}(\mathbf{\lambda})\mathbf{V}$, then we could also write $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$, where $\mathbf{V}$ is our encoding matrix.

This now leaves us with two separate but equivalent ways of computing our encoding matrix --- SVD or eigendecomposition.

## The decorrelation perspective

There are two key properties of PCA to understand: 
- Minimisation of the reconstruction error
- 'Decorrelation' of the features of data

We've just finished showing the first one so let's take a quick look at the second (much easier to show!).




Recall the unbiased estimator of the covariance of a centered matrix, $\mathbf{Z}$, is $\text{Var}[\mathbf{Z}] = \frac{1}{m-1}\mathbf{Z}^\top\mathbf{Z}$ where $m$ is the number of rows of the matrix. The covariance matrix is such that $\text{Var}(\mathbf{Z})_{i,j} = \text{cov}(\mathbf{Z}_i, \mathbf{Z}_j)$, that is it tells us the covariance between random variables.

If we can show that the covariance of PCA-encoded data, $\mathbf{Z}$ is a diagonal matrix, then we will know that covariance between each feature derived by PCA is zero.
We know that $\mathbf{Z} = \mathbf{X}\mathbf{D}$, if $\mathbf{X}$ is our original data and $\mathbf{D}$ is the encoding matrix.

Therefore $\text{Var}[\mathbf{Z}] = \frac{1}{m-1}\mathbf{Z}^\top\mathbf{Z} = \frac{1}{m-1}\mathbf{D}^\top\mathbf{X}^\top\mathbf{X}\mathbf{D}$.



However, recall that from the SVD approach we know that $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{D}^\top$ for a diagonal matrix $\mathbf{\Sigma}$.

From this we can rewrite $\mathbf{X}^\top\mathbf{X}$ as $(\mathbf{U}\mathbf{\Sigma}\mathbf{D}^\top)^\top(\mathbf{U}\mathbf{\Sigma}\mathbf{D}^\top)=\mathbf{D}\mathbf{\Sigma}^\top\mathbf{U}^\top\mathbf{U}\mathbf{\Sigma}\mathbf{D}^\top$, but since SVD gives us orthogonal matrices we know that $\mathbf{U}^\top\mathbf{U}=I$. Hence, $\mathbf{X}^\top\mathbf{X} = \mathbf{D}\mathbf{\Sigma}^\top\mathbf{\Sigma}\mathbf{D}^\top$

At this point, we're very close and you might be tempted to suggest that $\mathbf{\Sigma}^\top\mathbf{\Sigma} = \mathbf{\Sigma}^2$ since $\mathbf{\Sigma}$ is a diagonal matrix, and the transpose of a square, diagonal matrix is itself. However, $\mathbf{\Sigma}$ is not necessarily square! In fact, from the properties of SVD it has the same dimensions as $\mathbf{X}$. We do, however, know that the matrix $\mathbf{\Sigma}^\top\mathbf{\Sigma}$ is necessarily diagonal since it is a product of two diagonal matrices, and so the point still stands.

Hence, we have shown that the resulting encoding has a diagonal covariance matrix, which means that the features have no linear dependence between each other. We'll be able to visualise this later!

It's crucial to understand that a zero covariance does not imply that no relationship exists between the features of the transformed space but rather that no linear relationship can exist. The stronger condition of having no relationship, would require a different algorithm and is the approach taken by Independent Component Analysis.

One of the beauties of PCA is that there are lots of ways to think about it. I've only presented one, but you can also think about it as maximising variance, or transforming the coordinate axes amongst many others.

### Implementation

I didn't say much about what PCA can be used for, but now that we have an understanding of where the computation comes from, let's take a quick look at what it can be used for. We'll look at it as a technique for compression and for visualisation of high-dimensional data.

We can implement it in two-lines both via eigendecomposition and SVD. Although, we are essentially cheating since NumPy is doing all the heavy-lifting by computing the eigenvectors for us. That being said, computing eigenvectors should be itself the topic of a separate blog post.

The only nuance is that we need to _center_ the matrix. This means that each column is supposed to have a mean of zero. It is rather rare to see a description of why we need to do this, in fact, it will also very frequently work without doing it. However, recall that when we proved that PCA decorrelates data we made the assumption that the data was centered. Without this assumption we wouldn't have been able to use $\text{Var}[\mathbf{X}] = \frac{1}{m-1}\mathbf{Z}^\top\mathbf{Z}$ as an estimate of the covariance matrix.


```python
def pca_eig(X):
    X = center(X)
    eigenvals, eigenvectors = np.linalg.eig(X.T @ X)
    return eigenvals, eigenvectors

def pca_svd(X):
    X = center(X)
    _,root_eigenvalues,x = np.linalg.svd(X)
    return root_eigenvalues**2, x

def center(X):
    return X-np.mean(X, axis=0)
```

#### Toy example

We can generate a noisy straight line and take a look at the effect of PCA on it.


```python
def generate_noisy_line(gradient, N=100):
    X = np.zeros(shape=(N, 2))
    X[:,0] = np.random.uniform(-100, 100, size=N)
    X[:,1] = (X[:,0]*gradient)+ np.random.uniform(-100, 100, size=N)
    return X
```

Notice the direction of the eigenvectors plotted. This points out a property of PCA that was only briefly mentioned --- the PCA components are in the directions of maximum variance. PCA is trying to use its first component to capture as much of what distinguishes points as possible.
We can also see from the projected version the property of decorrelating data, which we just proved.


```python
X = generate_noisy_line(gradient=4)
_,eigenvectors = pca_eig(X)

# Use matrix multiplication '@' to find the projected versions
Z = X@eig_dec
```


![png](pca_files/pca_35_0.png)


Let's compare our simple two-line implementations to the sklearn implementation.


```python
# Our basic implementation vs. the sklearn implementation
from sklearn.decomposition import PCA
fitted = PCA().fit(center(X))
print(fitted.components_)
print(pca_eig(X)[1])
print(pca_svd(X)[1])
```

    [[ 0.22883214  0.9734659 ]
     [ 0.9734659  -0.22883214]]
    [[-0.9734659  -0.22883214]
     [ 0.22883214 -0.9734659 ]]
    [[-0.22883214 -0.9734659 ]
     [ 0.9734659  -0.22883214]]


They match (considering that eigenvectors may be arbitrarily multiplied by -1), but I still wouldn't rely on ours too much! SkLearn is doing a bunch of stuff behind the scenes to make sure it is always accurate!

#### Country data

Another common use-case of PCA is for visualising high-dimensional data. We can use the earlier derivation to select a matrix which projects our data down into two-dimensions. For example, I've taken some 7-dimensional data from [Wikipedia](https://en.wikipedia.org/wiki/World_Happiness_Report) on the happiness of populations around the world.
If we wished to plot the countries in relation to each other we could use the two eigenvectors with largest eigenvalues (as we showed earlier). This new matrix where each row only consists of two features, can be plotted and understood by mere mortals without the ability to visualise 7-dimensions.


```python
df = pd.read_csv("data/country-data.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.769</td>
      <td>1.340</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
      <td>0.153</td>
      <td>0.393</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Denmark</td>
      <td>7.600</td>
      <td>1.383</td>
      <td>1.573</td>
      <td>0.996</td>
      <td>0.592</td>
      <td>0.252</td>
      <td>0.410</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Norway</td>
      <td>7.554</td>
      <td>1.488</td>
      <td>1.582</td>
      <td>1.028</td>
      <td>0.603</td>
      <td>0.271</td>
      <td>0.341</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Iceland</td>
      <td>7.494</td>
      <td>1.380</td>
      <td>1.624</td>
      <td>1.026</td>
      <td>0.591</td>
      <td>0.354</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Netherlands</td>
      <td>7.488</td>
      <td>1.396</td>
      <td>1.522</td>
      <td>0.999</td>
      <td>0.557</td>
      <td>0.322</td>
      <td>0.298</td>
    </tr>
  </tbody>
</table>
</div>



Let's take the numeric columns (and exclude the score) which we will pass to the PCA algorithm. Given this, we take the two first columns of the returned eigenvector matrix (it's returned sorted by eigenvalue so this is ok!) and project down by taking the matrix multiplication with the encoding matrix.


```python
X = df[df.columns[3:]].to_numpy()
eigenvals, eigenvectors = pca_svd(X)
encoding_matrix = eigenvectors[:,:2]

# Sort in descending order
sorted_eigenvals = np.argsort(-1*eigenvals)
print(sorted_eigenvals)

projected = X@encoding_matrix
```

    [0 1 2 3 4 5 6 7]


We can now plot the projected version as a two-dimensional plot. Try zooming in and out to see how 'similar' countries tend to clump together in the resulting plot. Hover your mouse over a point to see additional information about the country. We've managed to preserve a lot of the information of the original 7 dimensions.





<div id="altair-viz-0e2fb404e1c949abb77a1ddefb129b5b"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-0e2fb404e1c949abb77a1ddefb129b5b") {
      outputDiv = document.getElementById("altair-viz-0e2fb404e1c949abb77a1ddefb129b5b");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "layer": [{"mark": {"type": "circle", "size": 60}, "encoding": {"color": {"type": "quantitative", "field": "Score", "scale": {"scheme": "plasma"}}, "tooltip": [{"type": "quantitative", "field": "Overall rank"}, {"type": "nominal", "field": "Country or region"}, {"type": "quantitative", "field": "Score"}, {"type": "quantitative", "field": "GDP per capita"}, {"type": "quantitative", "field": "Social support"}, {"type": "quantitative", "field": "Healthy life expectancy"}, {"type": "quantitative", "field": "Freedom to make life choices"}, {"type": "quantitative", "field": "Generosity"}, {"type": "quantitative", "field": "Perceptions of corruption"}, {"type": "quantitative", "field": "PCA Component 1"}, {"type": "quantitative", "field": "PCA Component 2"}], "x": {"type": "quantitative", "field": "PCA Component 1"}, "y": {"type": "quantitative", "field": "PCA Component 2"}}, "height": 500, "selection": {"selector044": {"type": "interval", "bind": "scales", "encodings": ["x", "y"]}}, "width": 1000}, {"mark": {"type": "text", "align": "left", "baseline": "middle", "color": "#000000", "dx": 7, "fill": "#000000", "fontSize": 8}, "encoding": {"color": {"type": "quantitative", "field": "Score", "scale": {"scheme": "plasma"}}, "text": {"type": "nominal", "field": "Country or region"}, "tooltip": [{"type": "quantitative", "field": "Overall rank"}, {"type": "nominal", "field": "Country or region"}, {"type": "quantitative", "field": "Score"}, {"type": "quantitative", "field": "GDP per capita"}, {"type": "quantitative", "field": "Social support"}, {"type": "quantitative", "field": "Healthy life expectancy"}, {"type": "quantitative", "field": "Freedom to make life choices"}, {"type": "quantitative", "field": "Generosity"}, {"type": "quantitative", "field": "Perceptions of corruption"}, {"type": "quantitative", "field": "PCA Component 1"}, {"type": "quantitative", "field": "PCA Component 2"}], "x": {"type": "quantitative", "field": "PCA Component 1"}, "y": {"type": "quantitative", "field": "PCA Component 2"}}, "height": 500, "width": 1000}], "data": {"name": "data-e74bd22965ffd033c3713925e90be719"}, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-e74bd22965ffd033c3713925e90be719": [{"Overall rank": 1, "Country or region": "Finland", "Score": 7.769, "GDP per capita": 1.34, "Social support": 1.587, "Healthy life expectancy": 0.986, "Freedom to make life choices": 0.596, "Generosity": 0.153, "Perceptions of corruption": 0.39299999999999996, "PCA Component 1": -1.307416536507356, "PCA Component 2": 0.7420367291478408}, {"Overall rank": 2, "Country or region": "Denmark", "Score": 7.6, "GDP per capita": 1.383, "Social support": 1.5730000000000002, "Healthy life expectancy": 0.996, "Freedom to make life choices": 0.5920000000000001, "Generosity": 0.252, "Perceptions of corruption": 0.41, "PCA Component 1": -1.2992361824062064, "PCA Component 2": 0.7313538354797929}, {"Overall rank": 3, "Country or region": "Norway", "Score": 7.553999999999999, "GDP per capita": 1.4880000000000002, "Social support": 1.5819999999999999, "Healthy life expectancy": 1.028, "Freedom to make life choices": 0.603, "Generosity": 0.271, "Perceptions of corruption": 0.341, "PCA Component 1": -1.356054273670667, "PCA Component 2": 0.6914273912769675}, {"Overall rank": 4, "Country or region": "Iceland", "Score": 7.494, "GDP per capita": 1.38, "Social support": 1.624, "Healthy life expectancy": 1.026, "Freedom to make life choices": 0.591, "Generosity": 0.354, "Perceptions of corruption": 0.11800000000000001, "PCA Component 1": -1.3161878697010283, "PCA Component 2": 0.8010128393635911}, {"Overall rank": 5, "Country or region": "Netherlands", "Score": 7.4879999999999995, "GDP per capita": 1.396, "Social support": 1.5219999999999998, "Healthy life expectancy": 0.9990000000000001, "Freedom to make life choices": 0.557, "Generosity": 0.322, "Perceptions of corruption": 0.298, "PCA Component 1": -1.2684541362252761, "PCA Component 2": 0.7020906326819389}, {"Overall rank": 6, "Country or region": "Switzerland", "Score": 7.48, "GDP per capita": 1.452, "Social support": 1.526, "Healthy life expectancy": 1.052, "Freedom to make life choices": 0.5720000000000001, "Generosity": 0.263, "Perceptions of corruption": 0.34299999999999997, "PCA Component 1": -1.3049022651068434, "PCA Component 2": 0.6752317059678119}, {"Overall rank": 7, "Country or region": "Sweden", "Score": 7.343, "GDP per capita": 1.3869999999999998, "Social support": 1.4869999999999999, "Healthy life expectancy": 1.0090000000000001, "Freedom to make life choices": 0.574, "Generosity": 0.267, "Perceptions of corruption": 0.373, "PCA Component 1": -1.2596329243667292, "PCA Component 2": 0.6706883136870164}, {"Overall rank": 8, "Country or region": "New Zealand", "Score": 7.307, "GDP per capita": 1.3030000000000002, "Social support": 1.557, "Healthy life expectancy": 1.026, "Freedom to make life choices": 0.585, "Generosity": 0.33, "Perceptions of corruption": 0.38, "PCA Component 1": -1.2340569039508549, "PCA Component 2": 0.789056204426521}, {"Overall rank": 9, "Country or region": "Canada", "Score": 7.278, "GDP per capita": 1.365, "Social support": 1.505, "Healthy life expectancy": 1.0390000000000001, "Freedom to make life choices": 0.584, "Generosity": 0.285, "Perceptions of corruption": 0.308, "PCA Component 1": -1.256981426316975, "PCA Component 2": 0.7081308126406685}, {"Overall rank": 10, "Country or region": "Austria", "Score": 7.246, "GDP per capita": 1.376, "Social support": 1.475, "Healthy life expectancy": 1.016, "Freedom to make life choices": 0.532, "Generosity": 0.244, "Perceptions of corruption": 0.226, "PCA Component 1": -1.2514975630676217, "PCA Component 2": 0.666335657785461}, {"Overall rank": 11, "Country or region": "Australia", "Score": 7.228, "GDP per capita": 1.3719999999999999, "Social support": 1.548, "Healthy life expectancy": 1.036, "Freedom to make life choices": 0.557, "Generosity": 0.332, "Perceptions of corruption": 0.29, "PCA Component 1": -1.2612537523349907, "PCA Component 2": 0.74782444496352}, {"Overall rank": 12, "Country or region": "Costa Rica", "Score": 7.167000000000001, "GDP per capita": 1.034, "Social support": 1.4409999999999998, "Healthy life expectancy": 0.963, "Freedom to make life choices": 0.5579999999999999, "Generosity": 0.14400000000000002, "Perceptions of corruption": 0.09300000000000001, "PCA Component 1": -1.1136377620952804, "PCA Component 2": 0.7938596349349472}, {"Overall rank": 13, "Country or region": "Israel", "Score": 7.138999999999999, "GDP per capita": 1.276, "Social support": 1.455, "Healthy life expectancy": 1.0290000000000001, "Freedom to make life choices": 0.371, "Generosity": 0.261, "Perceptions of corruption": 0.08199999999999999, "PCA Component 1": -1.1496488856462899, "PCA Component 2": 0.7224384976687883}, {"Overall rank": 14, "Country or region": "Luxembourg", "Score": 7.09, "GDP per capita": 1.609, "Social support": 1.479, "Healthy life expectancy": 1.012, "Freedom to make life choices": 0.526, "Generosity": 0.19399999999999998, "Perceptions of corruption": 0.316, "PCA Component 1": -1.3701913396106673, "PCA Component 2": 0.527829940034257}, {"Overall rank": 15, "Country or region": "United Kingdom", "Score": 7.053999999999999, "GDP per capita": 1.3330000000000002, "Social support": 1.538, "Healthy life expectancy": 0.996, "Freedom to make life choices": 0.45, "Generosity": 0.348, "Perceptions of corruption": 0.278, "PCA Component 1": -1.2066119083468807, "PCA Component 2": 0.7601792430021399}, {"Overall rank": 16, "Country or region": "Ireland", "Score": 7.021, "GDP per capita": 1.499, "Social support": 1.5530000000000002, "Healthy life expectancy": 0.9990000000000001, "Freedom to make life choices": 0.516, "Generosity": 0.298, "Perceptions of corruption": 0.31, "PCA Component 1": -1.3228505164191608, "PCA Component 2": 0.6653816931002713}, {"Overall rank": 17, "Country or region": "Germany", "Score": 6.985, "GDP per capita": 1.3730000000000002, "Social support": 1.454, "Healthy life expectancy": 0.987, "Freedom to make life choices": 0.495, "Generosity": 0.261, "Perceptions of corruption": 0.265, "PCA Component 1": -1.2268685641411134, "PCA Component 2": 0.649702457542653}, {"Overall rank": 18, "Country or region": "Belgium", "Score": 6.922999999999999, "GDP per capita": 1.3559999999999999, "Social support": 1.504, "Healthy life expectancy": 0.986, "Freedom to make life choices": 0.473, "Generosity": 0.16, "Perceptions of corruption": 0.21, "PCA Component 1": -1.2564029060373783, "PCA Component 2": 0.6767888991838462}, {"Overall rank": 19, "Country or region": "United States of America", "Score": 6.892, "GDP per capita": 1.433, "Social support": 1.4569999999999999, "Healthy life expectancy": 0.8740000000000001, "Freedom to make life choices": 0.45399999999999996, "Generosity": 0.28, "Perceptions of corruption": 0.128, "PCA Component 1": -1.2646475175677252, "PCA Component 2": 0.5928696576396235}, {"Overall rank": 20, "Country or region": "Czech Republic", "Score": 6.852, "GDP per capita": 1.2690000000000001, "Social support": 1.4869999999999999, "Healthy life expectancy": 0.92, "Freedom to make life choices": 0.457, "Generosity": 0.046, "Perceptions of corruption": 0.036000000000000004, "PCA Component 1": -1.2458614205289897, "PCA Component 2": 0.6700585738698432}, {"Overall rank": 21, "Country or region": "United Arab Emirates", "Score": 6.825, "GDP per capita": 1.5030000000000001, "Social support": 1.31, "Healthy life expectancy": 0.825, "Freedom to make life choices": 0.598, "Generosity": 0.262, "Perceptions of corruption": 0.182, "PCA Component 1": -1.2911355259617758, "PCA Component 2": 0.41570934105144247}, {"Overall rank": 22, "Country or region": "Malta", "Score": 6.726, "GDP per capita": 1.3, "Social support": 1.52, "Healthy life expectancy": 0.9990000000000001, "Freedom to make life choices": 0.564, "Generosity": 0.375, "Perceptions of corruption": 0.151, "PCA Component 1": -1.2229577811010097, "PCA Component 2": 0.7646094222122989}, {"Overall rank": 23, "Country or region": "Mexico", "Score": 6.595, "GDP per capita": 1.07, "Social support": 1.3230000000000002, "Healthy life expectancy": 0.861, "Freedom to make life choices": 0.433, "Generosity": 0.07400000000000001, "Perceptions of corruption": 0.073, "PCA Component 1": -1.072726086077528, "PCA Component 2": 0.6467493765761236}, {"Overall rank": 24, "Country or region": "France", "Score": 6.5920000000000005, "GDP per capita": 1.324, "Social support": 1.472, "Healthy life expectancy": 1.045, "Freedom to make life choices": 0.436, "Generosity": 0.111, "Perceptions of corruption": 0.183, "PCA Component 1": -1.2214742160067722, "PCA Component 2": 0.677919539400254}, {"Overall rank": 25, "Country or region": "Taiwan", "Score": 6.446000000000001, "GDP per capita": 1.368, "Social support": 1.43, "Healthy life expectancy": 0.914, "Freedom to make life choices": 0.35100000000000003, "Generosity": 0.242, "Perceptions of corruption": 0.09699999999999999, "PCA Component 1": -1.1948874221583436, "PCA Component 2": 0.6170318413630306}, {"Overall rank": 26, "Country or region": "Chile", "Score": 6.444, "GDP per capita": 1.159, "Social support": 1.369, "Healthy life expectancy": 0.92, "Freedom to make life choices": 0.35700000000000004, "Generosity": 0.187, "Perceptions of corruption": 0.055999999999999994, "PCA Component 1": -1.0827243023299222, "PCA Component 2": 0.6767056450978769}, {"Overall rank": 27, "Country or region": "Guatemala", "Score": 6.436, "GDP per capita": 0.8, "Social support": 1.2690000000000001, "Healthy life expectancy": 0.746, "Freedom to make life choices": 0.535, "Generosity": 0.175, "Perceptions of corruption": 0.078, "PCA Component 1": -0.9419192967129243, "PCA Component 2": 0.7407094215329233}, {"Overall rank": 28, "Country or region": "Saudi Arabia", "Score": 6.375, "GDP per capita": 1.403, "Social support": 1.357, "Healthy life expectancy": 0.795, "Freedom to make life choices": 0.439, "Generosity": 0.08, "Perceptions of corruption": 0.132, "PCA Component 1": -1.2547749030411273, "PCA Component 2": 0.4692647380956193}, {"Overall rank": 29, "Country or region": "Qatar", "Score": 6.374, "GDP per capita": 1.6840000000000002, "Social support": 1.3130000000000002, "Healthy life expectancy": 0.871, "Freedom to make life choices": 0.555, "Generosity": 0.22, "Perceptions of corruption": 0.16699999999999998, "PCA Component 1": -1.3728285422233195, "PCA Component 2": 0.3226671596299288}, {"Overall rank": 30, "Country or region": "Spain", "Score": 6.354, "GDP per capita": 1.286, "Social support": 1.484, "Healthy life expectancy": 1.062, "Freedom to make life choices": 0.36200000000000004, "Generosity": 0.153, "Perceptions of corruption": 0.079, "PCA Component 1": -1.181746670636251, "PCA Component 2": 0.7256879011228436}, {"Overall rank": 31, "Country or region": "Panama", "Score": 6.321000000000001, "GDP per capita": 1.149, "Social support": 1.442, "Healthy life expectancy": 0.91, "Freedom to make life choices": 0.516, "Generosity": 0.109, "Perceptions of corruption": 0.054000000000000006, "PCA Component 1": -1.1740130349984386, "PCA Component 2": 0.7102310957605936}, {"Overall rank": 32, "Country or region": "Brazil", "Score": 6.3, "GDP per capita": 1.004, "Social support": 1.439, "Healthy life expectancy": 0.802, "Freedom to make life choices": 0.39, "Generosity": 0.099, "Perceptions of corruption": 0.086, "PCA Component 1": -1.0734250588810066, "PCA Component 2": 0.7642495109937393}, {"Overall rank": 33, "Country or region": "Uruguay", "Score": 6.292999999999999, "GDP per capita": 1.124, "Social support": 1.465, "Healthy life expectancy": 0.8909999999999999, "Freedom to make life choices": 0.523, "Generosity": 0.127, "Perceptions of corruption": 0.15, "PCA Component 1": -1.164066660227803, "PCA Component 2": 0.740519201279125}, {"Overall rank": 34, "Country or region": "Singapore", "Score": 6.2620000000000005, "GDP per capita": 1.5719999999999998, "Social support": 1.463, "Healthy life expectancy": 1.141, "Freedom to make life choices": 0.556, "Generosity": 0.271, "Perceptions of corruption": 0.45299999999999996, "PCA Component 1": -1.3137703591599323, "PCA Component 2": 0.5872819897906791}, {"Overall rank": 35, "Country or region": "El Salvador", "Score": 6.252999999999999, "GDP per capita": 0.794, "Social support": 1.242, "Healthy life expectancy": 0.789, "Freedom to make life choices": 0.43, "Generosity": 0.09300000000000001, "Perceptions of corruption": 0.07400000000000001, "PCA Component 1": -0.9072537581545214, "PCA Component 2": 0.7234955461244511}, {"Overall rank": 36, "Country or region": "Italy", "Score": 6.223, "GDP per capita": 1.294, "Social support": 1.4880000000000002, "Healthy life expectancy": 1.0390000000000001, "Freedom to make life choices": 0.231, "Generosity": 0.158, "Perceptions of corruption": 0.03, "PCA Component 1": -1.1515557221797292, "PCA Component 2": 0.7259735478633783}, {"Overall rank": 37, "Country or region": "Bahrain", "Score": 6.199, "GDP per capita": 1.3619999999999999, "Social support": 1.368, "Healthy life expectancy": 0.871, "Freedom to make life choices": 0.536, "Generosity": 0.255, "Perceptions of corruption": 0.11, "PCA Component 1": -1.2266714347060939, "PCA Component 2": 0.5535579791889618}, {"Overall rank": 38, "Country or region": "Slovakia", "Score": 6.1979999999999995, "GDP per capita": 1.246, "Social support": 1.504, "Healthy life expectancy": 0.8809999999999999, "Freedom to make life choices": 0.33399999999999996, "Generosity": 0.121, "Perceptions of corruption": 0.013999999999999999, "PCA Component 1": -1.1935076099670912, "PCA Component 2": 0.7078712981945197}, {"Overall rank": 39, "Country or region": "Trinidad & Tobago", "Score": 6.192, "GDP per capita": 1.2309999999999999, "Social support": 1.4769999999999999, "Healthy life expectancy": 0.713, "Freedom to make life choices": 0.489, "Generosity": 0.185, "Perceptions of corruption": 0.016, "PCA Component 1": -1.230200556851635, "PCA Component 2": 0.6542442771460486}, {"Overall rank": 40, "Country or region": "Poland", "Score": 6.182, "GDP per capita": 1.206, "Social support": 1.4380000000000002, "Healthy life expectancy": 0.884, "Freedom to make life choices": 0.483, "Generosity": 0.11699999999999999, "Perceptions of corruption": 0.05, "PCA Component 1": -1.1919354039266878, "PCA Component 2": 0.6715416597924555}, {"Overall rank": 41, "Country or region": "Uzbekistan", "Score": 6.1739999999999995, "GDP per capita": 0.745, "Social support": 1.5290000000000001, "Healthy life expectancy": 0.7559999999999999, "Freedom to make life choices": 0.631, "Generosity": 0.322, "Perceptions of corruption": 0.24, "PCA Component 1": -1.004742699790786, "PCA Component 2": 1.000711170353329}, {"Overall rank": 42, "Country or region": "Lithuania", "Score": 6.149, "GDP per capita": 1.238, "Social support": 1.515, "Healthy life expectancy": 0.818, "Freedom to make life choices": 0.29100000000000004, "Generosity": 0.043, "Perceptions of corruption": 0.042, "PCA Component 1": -1.2013750106381378, "PCA Component 2": 0.6892503662555174}, {"Overall rank": 43, "Country or region": "Colombia", "Score": 6.125, "GDP per capita": 0.985, "Social support": 1.41, "Healthy life expectancy": 0.841, "Freedom to make life choices": 0.47, "Generosity": 0.099, "Perceptions of corruption": 0.034, "PCA Component 1": -1.077165809454316, "PCA Component 2": 0.7587783898874102}, {"Overall rank": 44, "Country or region": "Slovenia", "Score": 6.117999999999999, "GDP per capita": 1.258, "Social support": 1.5230000000000001, "Healthy life expectancy": 0.953, "Freedom to make life choices": 0.564, "Generosity": 0.14400000000000002, "Perceptions of corruption": 0.057, "PCA Component 1": -1.2629986631433086, "PCA Component 2": 0.7280853721034615}, {"Overall rank": 45, "Country or region": "Nicaragua", "Score": 6.105, "GDP per capita": 0.6940000000000001, "Social support": 1.325, "Healthy life expectancy": 0.835, "Freedom to make life choices": 0.435, "Generosity": 0.2, "Perceptions of corruption": 0.127, "PCA Component 1": -0.8611792126406439, "PCA Component 2": 0.8787333053526094}, {"Overall rank": 46, "Country or region": "Kosovo", "Score": 6.1, "GDP per capita": 0.882, "Social support": 1.232, "Healthy life expectancy": 0.758, "Freedom to make life choices": 0.489, "Generosity": 0.262, "Perceptions of corruption": 0.006, "PCA Component 1": -0.93980515058371, "PCA Component 2": 0.690429976979576}, {"Overall rank": 47, "Country or region": "Argentina", "Score": 6.086, "GDP per capita": 1.092, "Social support": 1.432, "Healthy life expectancy": 0.8809999999999999, "Freedom to make life choices": 0.47100000000000003, "Generosity": 0.066, "Perceptions of corruption": 0.05, "PCA Component 1": -1.1400779437093755, "PCA Component 2": 0.7197620222667654}, {"Overall rank": 48, "Country or region": "Romania", "Score": 6.07, "GDP per capita": 1.162, "Social support": 1.232, "Healthy life expectancy": 0.825, "Freedom to make life choices": 0.462, "Generosity": 0.083, "Perceptions of corruption": 0.005, "PCA Component 1": -1.0987319808393647, "PCA Component 2": 0.5154042504885139}, {"Overall rank": 49, "Country or region": "Cyprus", "Score": 6.046, "GDP per capita": 1.263, "Social support": 1.2229999999999999, "Healthy life expectancy": 1.042, "Freedom to make life choices": 0.406, "Generosity": 0.19, "Perceptions of corruption": 0.040999999999999995, "PCA Component 1": -1.077975431402804, "PCA Component 2": 0.5379180512122077}, {"Overall rank": 50, "Country or region": "Ecuador", "Score": 6.028, "GDP per capita": 0.912, "Social support": 1.3119999999999998, "Healthy life expectancy": 0.868, "Freedom to make life choices": 0.498, "Generosity": 0.126, "Perceptions of corruption": 0.087, "PCA Component 1": -0.9980875453735337, "PCA Component 2": 0.7363271765662579}, {"Overall rank": 51, "Country or region": "Kuwait", "Score": 6.011, "GDP per capita": 1.05, "Social support": 1.409, "Healthy life expectancy": 0.828, "Freedom to make life choices": 0.557, "Generosity": 0.359, "Perceptions of corruption": 0.027999999999999997, "PCA Component 1": -1.085300533878642, "PCA Component 2": 0.7686115828377399}, {"Overall rank": 52, "Country or region": "Thailand", "Score": 6.007999999999999, "GDP per capita": 1.05, "Social support": 1.409, "Healthy life expectancy": 0.828, "Freedom to make life choices": 0.557, "Generosity": 0.359, "Perceptions of corruption": 0.027999999999999997, "PCA Component 1": -1.085300533878642, "PCA Component 2": 0.7686115828377399}, {"Overall rank": 53, "Country or region": "Latvia", "Score": 5.94, "GDP per capita": 1.187, "Social support": 1.465, "Healthy life expectancy": 0.812, "Freedom to make life choices": 0.264, "Generosity": 0.075, "Perceptions of corruption": 0.064, "PCA Component 1": -1.1401994545478658, "PCA Component 2": 0.6861683887604823}, {"Overall rank": 54, "Country or region": "South Korea", "Score": 5.895, "GDP per capita": 1.301, "Social support": 1.219, "Healthy life expectancy": 1.036, "Freedom to make life choices": 0.159, "Generosity": 0.175, "Perceptions of corruption": 0.055999999999999994, "PCA Component 1": -1.0201343743380133, "PCA Component 2": 0.5223424027054868}, {"Overall rank": 55, "Country or region": "Estonia", "Score": 5.893, "GDP per capita": 1.237, "Social support": 1.528, "Healthy life expectancy": 0.8740000000000001, "Freedom to make life choices": 0.495, "Generosity": 0.10300000000000001, "Perceptions of corruption": 0.161, "PCA Component 1": -1.2422763800725796, "PCA Component 2": 0.7173648509049405}, {"Overall rank": 56, "Country or region": "Jamaica", "Score": 5.89, "GDP per capita": 0.831, "Social support": 1.4780000000000002, "Healthy life expectancy": 0.831, "Freedom to make life choices": 0.49, "Generosity": 0.107, "Perceptions of corruption": 0.027999999999999997, "PCA Component 1": -1.0342915101250805, "PCA Component 2": 0.8950917434152396}, {"Overall rank": 57, "Country or region": "Mauritius", "Score": 5.888, "GDP per capita": 1.12, "Social support": 1.402, "Healthy life expectancy": 0.7979999999999999, "Freedom to make life choices": 0.498, "Generosity": 0.215, "Perceptions of corruption": 0.06, "PCA Component 1": -1.128911298881524, "PCA Component 2": 0.6884178123371438}, {"Overall rank": 58, "Country or region": "Japan", "Score": 5.886, "GDP per capita": 1.327, "Social support": 1.419, "Healthy life expectancy": 1.088, "Freedom to make life choices": 0.445, "Generosity": 0.069, "Perceptions of corruption": 0.14, "PCA Component 1": -1.2116245356775468, "PCA Component 2": 0.6377874434429376}, {"Overall rank": 59, "Country or region": "Honduras", "Score": 5.86, "GDP per capita": 0.642, "Social support": 1.236, "Healthy life expectancy": 0.828, "Freedom to make life choices": 0.507, "Generosity": 0.24600000000000002, "Perceptions of corruption": 0.078, "PCA Component 1": -0.8178721968016435, "PCA Component 2": 0.8432192655518478}, {"Overall rank": 60, "Country or region": "Kazakhstan", "Score": 5.809, "GDP per capita": 1.173, "Social support": 1.508, "Healthy life expectancy": 0.7290000000000001, "Freedom to make life choices": 0.41, "Generosity": 0.146, "Perceptions of corruption": 0.096, "PCA Component 1": -1.188894977109298, "PCA Component 2": 0.7113439986900201}, {"Overall rank": 61, "Country or region": "Bolivia", "Score": 5.779, "GDP per capita": 0.7759999999999999, "Social support": 1.209, "Healthy life expectancy": 0.706, "Freedom to make life choices": 0.511, "Generosity": 0.13699999999999998, "Perceptions of corruption": 0.064, "PCA Component 1": -0.9120328921119433, "PCA Component 2": 0.6901836559626912}, {"Overall rank": 62, "Country or region": "Hungary", "Score": 5.757999999999999, "GDP per capita": 1.2009999999999998, "Social support": 1.41, "Healthy life expectancy": 0.828, "Freedom to make life choices": 0.19899999999999998, "Generosity": 0.081, "Perceptions of corruption": 0.02, "PCA Component 1": -1.1051482852081478, "PCA Component 2": 0.645169796376385}, {"Overall rank": 63, "Country or region": "Paraguay", "Score": 5.742999999999999, "GDP per capita": 0.855, "Social support": 1.475, "Healthy life expectancy": 0.777, "Freedom to make life choices": 0.514, "Generosity": 0.184, "Perceptions of corruption": 0.08, "PCA Component 1": -1.0390181323760614, "PCA Component 2": 0.8797385901937427}, {"Overall rank": 64, "Country or region": "Northern Cyprus", "Score": 5.718, "GDP per capita": 1.263, "Social support": 1.252, "Healthy life expectancy": 1.042, "Freedom to make life choices": 0.41700000000000004, "Generosity": 0.191, "Perceptions of corruption": 0.162, "PCA Component 1": -1.0833844845228413, "PCA Component 2": 0.560324058676173}, {"Overall rank": 65, "Country or region": "Peru", "Score": 5.697, "GDP per capita": 0.96, "Social support": 1.274, "Healthy life expectancy": 0.8540000000000001, "Freedom to make life choices": 0.455, "Generosity": 0.083, "Perceptions of corruption": 0.027000000000000003, "PCA Component 1": -1.0082246899384515, "PCA Component 2": 0.6693216521602908}, {"Overall rank": 66, "Country or region": "Portugal", "Score": 5.693, "GDP per capita": 1.2209999999999999, "Social support": 1.431, "Healthy life expectancy": 0.9990000000000001, "Freedom to make life choices": 0.508, "Generosity": 0.047, "Perceptions of corruption": 0.025, "PCA Component 1": -1.2073898039269322, "PCA Component 2": 0.6731401529435108}, {"Overall rank": 67, "Country or region": "Pakistan", "Score": 5.653, "GDP per capita": 0.677, "Social support": 0.8859999999999999, "Healthy life expectancy": 0.535, "Freedom to make life choices": 0.313, "Generosity": 0.22, "Perceptions of corruption": 0.098, "PCA Component 1": -0.6723108799950692, "PCA Component 2": 0.47906983744083964}, {"Overall rank": 68, "Country or region": "Russia", "Score": 5.648, "GDP per capita": 1.183, "Social support": 1.452, "Healthy life expectancy": 0.726, "Freedom to make life choices": 0.33399999999999996, "Generosity": 0.08199999999999999, "Perceptions of corruption": 0.031, "PCA Component 1": -1.1660808982263586, "PCA Component 2": 0.6521607217928868}, {"Overall rank": 69, "Country or region": "Philippines", "Score": 5.631, "GDP per capita": 0.807, "Social support": 1.2930000000000001, "Healthy life expectancy": 0.657, "Freedom to make life choices": 0.5579999999999999, "Generosity": 0.11699999999999999, "Perceptions of corruption": 0.107, "PCA Component 1": -0.9819758055150904, "PCA Component 2": 0.7171225989211191}, {"Overall rank": 70, "Country or region": "Serbia", "Score": 5.603, "GDP per capita": 1.004, "Social support": 1.383, "Healthy life expectancy": 0.8540000000000001, "Freedom to make life choices": 0.282, "Generosity": 0.13699999999999998, "Perceptions of corruption": 0.039, "PCA Component 1": -1.0071021583005004, "PCA Component 2": 0.7494459675221233}, {"Overall rank": 71, "Country or region": "Moldova", "Score": 5.529, "GDP per capita": 0.685, "Social support": 1.328, "Healthy life expectancy": 0.7390000000000001, "Freedom to make life choices": 0.245, "Generosity": 0.18100000000000002, "Perceptions of corruption": 0.0, "PCA Component 1": -0.8229756219999443, "PCA Component 2": 0.865388618679036}, {"Overall rank": 72, "Country or region": "Libya", "Score": 5.525, "GDP per capita": 1.044, "Social support": 1.3030000000000002, "Healthy life expectancy": 0.6729999999999999, "Freedom to make life choices": 0.41600000000000004, "Generosity": 0.133, "Perceptions of corruption": 0.152, "PCA Component 1": -1.050040696185121, "PCA Component 2": 0.6078565833090793}, {"Overall rank": 73, "Country or region": "Montenegro", "Score": 5.523, "GDP per capita": 1.051, "Social support": 1.361, "Healthy life expectancy": 0.871, "Freedom to make life choices": 0.19699999999999998, "Generosity": 0.142, "Perceptions of corruption": 0.08, "PCA Component 1": -0.9887300008730608, "PCA Component 2": 0.7168340165152872}, {"Overall rank": 74, "Country or region": "Tajikistan", "Score": 5.4670000000000005, "GDP per capita": 0.493, "Social support": 1.0979999999999999, "Healthy life expectancy": 0.718, "Freedom to make life choices": 0.389, "Generosity": 0.23, "Perceptions of corruption": 0.14400000000000002, "PCA Component 1": -0.6628727017265642, "PCA Component 2": 0.7936844694178234}, {"Overall rank": 75, "Country or region": "Croatia", "Score": 5.432, "GDP per capita": 1.155, "Social support": 1.266, "Healthy life expectancy": 0.914, "Freedom to make life choices": 0.29600000000000004, "Generosity": 0.11900000000000001, "Perceptions of corruption": 0.022000000000000002, "PCA Component 1": -1.0377327328478267, "PCA Component 2": 0.5869019379542209}, {"Overall rank": 76, "Country or region": "Hong Kong", "Score": 5.43, "GDP per capita": 1.4380000000000002, "Social support": 1.2770000000000001, "Healthy life expectancy": 1.122, "Freedom to make life choices": 0.44, "Generosity": 0.258, "Perceptions of corruption": 0.287, "PCA Component 1": -1.1546341878190487, "PCA Component 2": 0.5171493171249907}, {"Overall rank": 77, "Country or region": "Dominican Republic", "Score": 5.425, "GDP per capita": 1.015, "Social support": 1.401, "Healthy life expectancy": 0.779, "Freedom to make life choices": 0.49700000000000005, "Generosity": 0.113, "Perceptions of corruption": 0.10099999999999999, "PCA Component 1": -1.0960024464461244, "PCA Component 2": 0.7197177854404276}, {"Overall rank": 78, "Country or region": "Bosnia and Herzegovina", "Score": 5.386, "GDP per capita": 0.945, "Social support": 1.212, "Healthy life expectancy": 0.845, "Freedom to make life choices": 0.212, "Generosity": 0.263, "Perceptions of corruption": 0.006, "PCA Component 1": -0.8657385175318122, "PCA Component 2": 0.6792559421104843}, {"Overall rank": 79, "Country or region": "Turkey", "Score": 5.372999999999999, "GDP per capita": 1.183, "Social support": 1.36, "Healthy life expectancy": 0.8079999999999999, "Freedom to make life choices": 0.195, "Generosity": 0.083, "Perceptions of corruption": 0.106, "PCA Component 1": -1.0702893573420886, "PCA Component 2": 0.6123513342889215}, {"Overall rank": 80, "Country or region": "Malaysia", "Score": 5.3389999999999995, "GDP per capita": 1.2209999999999999, "Social support": 1.171, "Healthy life expectancy": 0.828, "Freedom to make life choices": 0.508, "Generosity": 0.26, "Perceptions of corruption": 0.024, "PCA Component 1": -1.0802372245650351, "PCA Component 2": 0.4716159491238152}, {"Overall rank": 81, "Country or region": "Belarus", "Score": 5.3229999999999995, "GDP per capita": 1.067, "Social support": 1.465, "Healthy life expectancy": 0.789, "Freedom to make life choices": 0.235, "Generosity": 0.094, "Perceptions of corruption": 0.142, "PCA Component 1": -1.0644093595437618, "PCA Component 2": 0.7530250545144341}, {"Overall rank": 82, "Country or region": "Greece", "Score": 5.287000000000001, "GDP per capita": 1.181, "Social support": 1.156, "Healthy life expectancy": 0.9990000000000001, "Freedom to make life choices": 0.067, "Generosity": 0.0, "Perceptions of corruption": 0.034, "PCA Component 1": -0.9483369563193087, "PCA Component 2": 0.4986271190032733}, {"Overall rank": 83, "Country or region": "Mongolia", "Score": 5.285, "GDP per capita": 0.948, "Social support": 1.531, "Healthy life expectancy": 0.667, "Freedom to make life choices": 0.317, "Generosity": 0.235, "Perceptions of corruption": 0.038, "PCA Component 1": -1.0509651067893897, "PCA Component 2": 0.8617557806964532}, {"Overall rank": 84, "Country or region": "North Macedonia", "Score": 5.274, "GDP per capita": 0.983, "Social support": 1.294, "Healthy life expectancy": 0.838, "Freedom to make life choices": 0.345, "Generosity": 0.185, "Perceptions of corruption": 0.034, "PCA Component 1": -0.9735235972761835, "PCA Component 2": 0.6951898686921004}, {"Overall rank": 85, "Country or region": "Nigeria", "Score": 5.265, "GDP per capita": 0.696, "Social support": 1.111, "Healthy life expectancy": 0.245, "Freedom to make life choices": 0.426, "Generosity": 0.215, "Perceptions of corruption": 0.040999999999999995, "PCA Component 1": -0.8456565406115546, "PCA Component 2": 0.5535430521301765}, {"Overall rank": 86, "Country or region": "Kyrgyzstan", "Score": 5.261, "GDP per capita": 0.551, "Social support": 1.4380000000000002, "Healthy life expectancy": 0.723, "Freedom to make life choices": 0.508, "Generosity": 0.3, "Perceptions of corruption": 0.023, "PCA Component 1": -0.8589411996991139, "PCA Component 2": 1.0315220955376754}, {"Overall rank": 87, "Country or region": "Turkmenistan", "Score": 5.247000000000001, "GDP per capita": 1.052, "Social support": 1.538, "Healthy life expectancy": 0.657, "Freedom to make life choices": 0.39399999999999996, "Generosity": 0.244, "Perceptions of corruption": 0.027999999999999997, "PCA Component 1": -1.1295281636875976, "PCA Component 2": 0.8037451435811628}, {"Overall rank": 88, "Country or region": "Algeria", "Score": 5.211, "GDP per capita": 1.002, "Social support": 1.16, "Healthy life expectancy": 0.785, "Freedom to make life choices": 0.086, "Generosity": 0.073, "Perceptions of corruption": 0.114, "PCA Component 1": -0.8708564217063239, "PCA Component 2": 0.5577066672444704}, {"Overall rank": 89, "Country or region": "Morocco", "Score": 5.207999999999999, "GDP per capita": 0.8009999999999999, "Social support": 0.782, "Healthy life expectancy": 0.782, "Freedom to make life choices": 0.418, "Generosity": 0.036000000000000004, "Perceptions of corruption": 0.076, "PCA Component 1": -0.7360267083425196, "PCA Component 2": 0.3529087086684145}, {"Overall rank": 90, "Country or region": "Azerbaijan", "Score": 5.207999999999999, "GDP per capita": 1.043, "Social support": 1.147, "Healthy life expectancy": 0.769, "Freedom to make life choices": 0.35100000000000003, "Generosity": 0.035, "Perceptions of corruption": 0.182, "PCA Component 1": -0.9735718235272689, "PCA Component 2": 0.4979642000892257}, {"Overall rank": 91, "Country or region": "Lebanon", "Score": 5.197, "GDP per capita": 0.987, "Social support": 1.224, "Healthy life expectancy": 0.815, "Freedom to make life choices": 0.21600000000000003, "Generosity": 0.166, "Perceptions of corruption": 0.027000000000000003, "PCA Component 1": -0.9141407284872503, "PCA Component 2": 0.6358470417541202}, {"Overall rank": 92, "Country or region": "Indonesia", "Score": 5.192, "GDP per capita": 0.9309999999999999, "Social support": 1.203, "Healthy life expectancy": 0.66, "Freedom to make life choices": 0.491, "Generosity": 0.498, "Perceptions of corruption": 0.027999999999999997, "PCA Component 1": -0.9146505755447247, "PCA Component 2": 0.6639130646905239}, {"Overall rank": 93, "Country or region": "China", "Score": 5.191, "GDP per capita": 1.0290000000000001, "Social support": 1.125, "Healthy life expectancy": 0.893, "Freedom to make life choices": 0.521, "Generosity": 0.057999999999999996, "Perceptions of corruption": 0.1, "PCA Component 1": -0.99872512195915, "PCA Component 2": 0.5183825885013371}, {"Overall rank": 94, "Country or region": "Vietnam", "Score": 5.175, "GDP per capita": 0.741, "Social support": 1.3459999999999999, "Healthy life expectancy": 0.851, "Freedom to make life choices": 0.5429999999999999, "Generosity": 0.147, "Perceptions of corruption": 0.073, "PCA Component 1": -0.9398967401620258, "PCA Component 2": 0.8554824249160005}, {"Overall rank": 95, "Country or region": "Bhutan", "Score": 5.082, "GDP per capita": 0.813, "Social support": 1.321, "Healthy life expectancy": 0.604, "Freedom to make life choices": 0.457, "Generosity": 0.37, "Perceptions of corruption": 0.16699999999999998, "PCA Component 1": -0.9142679785211552, "PCA Component 2": 0.7803543482628682}, {"Overall rank": 96, "Country or region": "Cameroon", "Score": 5.044, "GDP per capita": 0.5489999999999999, "Social support": 0.91, "Healthy life expectancy": 0.331, "Freedom to make life choices": 0.381, "Generosity": 0.187, "Perceptions of corruption": 0.037000000000000005, "PCA Component 1": -0.6748222290588983, "PCA Component 2": 0.5017983782288842}, {"Overall rank": 97, "Country or region": "Bulgaria", "Score": 5.011, "GDP per capita": 1.092, "Social support": 1.5130000000000001, "Healthy life expectancy": 0.815, "Freedom to make life choices": 0.311, "Generosity": 0.081, "Perceptions of corruption": 0.004, "PCA Component 1": -1.1301543953915973, "PCA Component 2": 0.7755156974276498}, {"Overall rank": 98, "Country or region": "Ghana", "Score": 4.996, "GDP per capita": 0.611, "Social support": 0.868, "Healthy life expectancy": 0.486, "Freedom to make life choices": 0.381, "Generosity": 0.245, "Perceptions of corruption": 0.04, "PCA Component 1": -0.6588998606304612, "PCA Component 2": 0.4900202776689034}, {"Overall rank": 99, "Country or region": "Ivory Coast", "Score": 4.944, "GDP per capita": 0.569, "Social support": 0.8079999999999999, "Healthy life expectancy": 0.23199999999999998, "Freedom to make life choices": 0.35200000000000004, "Generosity": 0.154, "Perceptions of corruption": 0.09, "PCA Component 1": -0.6489902359144378, "PCA Component 2": 0.37969627344999224}, {"Overall rank": 100, "Country or region": "Nepal", "Score": 4.913, "GDP per capita": 0.446, "Social support": 1.226, "Healthy life expectancy": 0.677, "Freedom to make life choices": 0.439, "Generosity": 0.285, "Perceptions of corruption": 0.08900000000000001, "PCA Component 1": -0.7040844775673457, "PCA Component 2": 0.9157161627497329}, {"Overall rank": 101, "Country or region": "Jordan", "Score": 4.906000000000001, "GDP per capita": 0.8370000000000001, "Social support": 1.225, "Healthy life expectancy": 0.815, "Freedom to make life choices": 0.38299999999999995, "Generosity": 0.11, "Perceptions of corruption": 0.13, "PCA Component 1": -0.896192919344152, "PCA Component 2": 0.6999860203251884}, {"Overall rank": 102, "Country or region": "Benin", "Score": 4.883, "GDP per capita": 0.39299999999999996, "Social support": 0.43700000000000006, "Healthy life expectancy": 0.397, "Freedom to make life choices": 0.349, "Generosity": 0.175, "Perceptions of corruption": 0.08199999999999999, "PCA Component 1": -0.3905449149536203, "PCA Component 2": 0.2436000882177493}, {"Overall rank": 103, "Country or region": "Congo (Brazzaville)", "Score": 4.812, "GDP per capita": 0.6729999999999999, "Social support": 0.799, "Healthy life expectancy": 0.508, "Freedom to make life choices": 0.37200000000000005, "Generosity": 0.105, "Perceptions of corruption": 0.09300000000000001, "PCA Component 1": -0.6811474217361815, "PCA Component 2": 0.37926301479637076}, {"Overall rank": 104, "Country or region": "Gabon", "Score": 4.7989999999999995, "GDP per capita": 1.057, "Social support": 1.183, "Healthy life expectancy": 0.5710000000000001, "Freedom to make life choices": 0.295, "Generosity": 0.043, "Perceptions of corruption": 0.055, "PCA Component 1": -1.0082557856670509, "PCA Component 2": 0.46727777732050824}, {"Overall rank": 105, "Country or region": "Laos", "Score": 4.796, "GDP per capita": 0.764, "Social support": 1.03, "Healthy life expectancy": 0.551, "Freedom to make life choices": 0.547, "Generosity": 0.266, "Perceptions of corruption": 0.16399999999999998, "PCA Component 1": -0.8299361055626928, "PCA Component 2": 0.5425642517203562}, {"Overall rank": 106, "Country or region": "South Africa", "Score": 4.7219999999999995, "GDP per capita": 0.96, "Social support": 1.351, "Healthy life expectancy": 0.469, "Freedom to make life choices": 0.389, "Generosity": 0.13, "Perceptions of corruption": 0.055, "PCA Component 1": -1.0507125069298255, "PCA Component 2": 0.6358234728064366}, {"Overall rank": 107, "Country or region": "Albania", "Score": 4.718999999999999, "GDP per capita": 0.9470000000000001, "Social support": 0.848, "Healthy life expectancy": 0.8740000000000001, "Freedom to make life choices": 0.38299999999999995, "Generosity": 0.17800000000000002, "Perceptions of corruption": 0.027000000000000003, "PCA Component 1": -0.7878701862986041, "PCA Component 2": 0.3791977470036081}, {"Overall rank": 108, "Country or region": "Venezuela", "Score": 4.707, "GDP per capita": 0.96, "Social support": 1.4269999999999998, "Healthy life expectancy": 0.805, "Freedom to make life choices": 0.154, "Generosity": 0.064, "Perceptions of corruption": 0.047, "PCA Component 1": -0.9824964797095735, "PCA Component 2": 0.7858548084359198}, {"Overall rank": 109, "Country or region": "Cambodia", "Score": 4.7, "GDP per capita": 0.574, "Social support": 1.122, "Healthy life expectancy": 0.637, "Freedom to make life choices": 0.609, "Generosity": 0.23199999999999998, "Perceptions of corruption": 0.062, "PCA Component 1": -0.7969568180874514, "PCA Component 2": 0.7323280356060231}, {"Overall rank": 110, "Country or region": "Palestinian Territories", "Score": 4.696000000000001, "GDP per capita": 0.657, "Social support": 1.247, "Healthy life expectancy": 0.672, "Freedom to make life choices": 0.225, "Generosity": 0.10300000000000001, "Perceptions of corruption": 0.066, "PCA Component 1": -0.7890477068111058, "PCA Component 2": 0.7851187260535271}, {"Overall rank": 111, "Country or region": "Senegal", "Score": 4.681, "GDP per capita": 0.45, "Social support": 1.1340000000000001, "Healthy life expectancy": 0.5710000000000001, "Freedom to make life choices": 0.292, "Generosity": 0.153, "Perceptions of corruption": 0.07200000000000001, "PCA Component 1": -0.6636257395185639, "PCA Component 2": 0.7934018961415771}, {"Overall rank": 112, "Country or region": "Somalia", "Score": 4.668, "GDP per capita": 0.0, "Social support": 0.698, "Healthy life expectancy": 0.268, "Freedom to make life choices": 0.5589999999999999, "Generosity": 0.243, "Perceptions of corruption": 0.27, "PCA Component 1": -0.3523660963443854, "PCA Component 2": 0.6323285236381049}, {"Overall rank": 113, "Country or region": "Namibia", "Score": 4.638999999999999, "GDP per capita": 0.879, "Social support": 1.3130000000000002, "Healthy life expectancy": 0.47700000000000004, "Freedom to make life choices": 0.401, "Generosity": 0.07, "Perceptions of corruption": 0.055999999999999994, "PCA Component 1": -1.0104794233148655, "PCA Component 2": 0.6407887947964067}, {"Overall rank": 114, "Country or region": "Niger", "Score": 4.628, "GDP per capita": 0.138, "Social support": 0.774, "Healthy life expectancy": 0.366, "Freedom to make life choices": 0.318, "Generosity": 0.188, "Perceptions of corruption": 0.102, "PCA Component 1": -0.388147257604198, "PCA Component 2": 0.641250966788732}, {"Overall rank": 115, "Country or region": "Burkina Faso", "Score": 4.587, "GDP per capita": 0.331, "Social support": 1.056, "Healthy life expectancy": 0.38, "Freedom to make life choices": 0.255, "Generosity": 0.177, "Perceptions of corruption": 0.113, "PCA Component 1": -0.575947210021357, "PCA Component 2": 0.7547161339686337}, {"Overall rank": 116, "Country or region": "Armenia", "Score": 4.559, "GDP per capita": 0.85, "Social support": 1.055, "Healthy life expectancy": 0.815, "Freedom to make life choices": 0.28300000000000003, "Generosity": 0.095, "Perceptions of corruption": 0.064, "PCA Component 1": -0.8116769328539521, "PCA Component 2": 0.5641653066257902}, {"Overall rank": 117, "Country or region": "Iran", "Score": 4.548, "GDP per capita": 1.1, "Social support": 0.8420000000000001, "Healthy life expectancy": 0.785, "Freedom to make life choices": 0.305, "Generosity": 0.27, "Perceptions of corruption": 0.125, "PCA Component 1": -0.8205964847515663, "PCA Component 2": 0.2886005290748393}, {"Overall rank": 118, "Country or region": "Guinea", "Score": 4.534, "GDP per capita": 0.38, "Social support": 0.8290000000000001, "Healthy life expectancy": 0.375, "Freedom to make life choices": 0.332, "Generosity": 0.207, "Perceptions of corruption": 0.086, "PCA Component 1": -0.5305716543625755, "PCA Component 2": 0.5536060199223747}, {"Overall rank": 119, "Country or region": "Georgia", "Score": 4.519, "GDP per capita": 0.8859999999999999, "Social support": 0.6659999999999999, "Healthy life expectancy": 0.752, "Freedom to make life choices": 0.34600000000000003, "Generosity": 0.043, "Perceptions of corruption": 0.16399999999999998, "PCA Component 1": -0.7044720458359317, "PCA Component 2": 0.21374966932022113}, {"Overall rank": 120, "Country or region": "Gambia", "Score": 4.516, "GDP per capita": 0.308, "Social support": 0.9390000000000001, "Healthy life expectancy": 0.428, "Freedom to make life choices": 0.382, "Generosity": 0.26899999999999996, "Perceptions of corruption": 0.16699999999999998, "PCA Component 1": -0.5294379555993587, "PCA Component 2": 0.7038829907009128}, {"Overall rank": 121, "Country or region": "Kenya", "Score": 4.5089999999999995, "GDP per capita": 0.512, "Social support": 0.983, "Healthy life expectancy": 0.581, "Freedom to make life choices": 0.431, "Generosity": 0.37200000000000005, "Perceptions of corruption": 0.053, "PCA Component 1": -0.6335997224992578, "PCA Component 2": 0.6844283475648623}, {"Overall rank": 122, "Country or region": "Mauritania", "Score": 4.49, "GDP per capita": 0.57, "Social support": 1.167, "Healthy life expectancy": 0.489, "Freedom to make life choices": 0.066, "Generosity": 0.106, "Perceptions of corruption": 0.08800000000000001, "PCA Component 1": -0.6828951693169136, "PCA Component 2": 0.7313264908072125}, {"Overall rank": 123, "Country or region": "Mozambique", "Score": 4.466, "GDP per capita": 0.204, "Social support": 0.986, "Healthy life expectancy": 0.39, "Freedom to make life choices": 0.494, "Generosity": 0.19699999999999998, "Perceptions of corruption": 0.138, "PCA Component 1": -0.5531686844774658, "PCA Component 2": 0.7661115199350288}, {"Overall rank": 124, "Country or region": "Tunisia", "Score": 4.461, "GDP per capita": 0.9209999999999999, "Social support": 1.0, "Healthy life expectancy": 0.815, "Freedom to make life choices": 0.16699999999999998, "Generosity": 0.059000000000000004, "Perceptions of corruption": 0.055, "PCA Component 1": -0.796487092803028, "PCA Component 2": 0.48081463892279863}, {"Overall rank": 125, "Country or region": "Bangladesh", "Score": 4.456, "GDP per capita": 0.562, "Social support": 0.9279999999999999, "Healthy life expectancy": 0.723, "Freedom to make life choices": 0.527, "Generosity": 0.166, "Perceptions of corruption": 0.14300000000000002, "PCA Component 1": -0.6852426325958049, "PCA Component 2": 0.6046304883802999}, {"Overall rank": 126, "Country or region": "Iraq", "Score": 4.437, "GDP per capita": 1.043, "Social support": 0.98, "Healthy life expectancy": 0.574, "Freedom to make life choices": 0.24100000000000002, "Generosity": 0.14800000000000002, "Perceptions of corruption": 0.08900000000000001, "PCA Component 1": -0.8791468848244867, "PCA Component 2": 0.34555573932256145}, {"Overall rank": 127, "Country or region": "Congo (Kinshasa)", "Score": 4.418, "GDP per capita": 0.094, "Social support": 1.125, "Healthy life expectancy": 0.35700000000000004, "Freedom to make life choices": 0.26899999999999996, "Generosity": 0.212, "Perceptions of corruption": 0.053, "PCA Component 1": -0.4906915418407393, "PCA Component 2": 0.9405543723076446}, {"Overall rank": 128, "Country or region": "Mali", "Score": 4.39, "GDP per capita": 0.385, "Social support": 1.105, "Healthy life expectancy": 0.308, "Freedom to make life choices": 0.327, "Generosity": 0.153, "Perceptions of corruption": 0.052000000000000005, "PCA Component 1": -0.6627307751845617, "PCA Component 2": 0.7328557871509733}, {"Overall rank": 129, "Country or region": "Sierra Leone", "Score": 4.374, "GDP per capita": 0.268, "Social support": 0.841, "Healthy life expectancy": 0.242, "Freedom to make life choices": 0.309, "Generosity": 0.252, "Perceptions of corruption": 0.045, "PCA Component 1": -0.48201055711884927, "PCA Component 2": 0.5994903496431583}, {"Overall rank": 130, "Country or region": "Sri Lanka", "Score": 4.3660000000000005, "GDP per capita": 0.9490000000000001, "Social support": 1.265, "Healthy life expectancy": 0.831, "Freedom to make life choices": 0.47, "Generosity": 0.244, "Perceptions of corruption": 0.047, "PCA Component 1": -0.972217899475522, "PCA Component 2": 0.6957956472882859}, {"Overall rank": 131, "Country or region": "Myanmar", "Score": 4.36, "GDP per capita": 0.71, "Social support": 1.181, "Healthy life expectancy": 0.555, "Freedom to make life choices": 0.525, "Generosity": 0.5660000000000001, "Perceptions of corruption": 0.172, "PCA Component 1": -0.7942455602524906, "PCA Component 2": 0.7550679556928841}, {"Overall rank": 132, "Country or region": "Chad", "Score": 4.35, "GDP per capita": 0.35, "Social support": 0.7659999999999999, "Healthy life expectancy": 0.192, "Freedom to make life choices": 0.174, "Generosity": 0.198, "Perceptions of corruption": 0.078, "PCA Component 1": -0.4644896633617454, "PCA Component 2": 0.47819132672454046}, {"Overall rank": 133, "Country or region": "Ukraine", "Score": 4.332, "GDP per capita": 0.82, "Social support": 1.39, "Healthy life expectancy": 0.7390000000000001, "Freedom to make life choices": 0.17800000000000002, "Generosity": 0.187, "Perceptions of corruption": 0.01, "PCA Component 1": -0.8914292850113621, "PCA Component 2": 0.8423723928974542}, {"Overall rank": 134, "Country or region": "Ethiopia", "Score": 4.2860000000000005, "GDP per capita": 0.336, "Social support": 1.033, "Healthy life expectancy": 0.532, "Freedom to make life choices": 0.344, "Generosity": 0.209, "Perceptions of corruption": 0.1, "PCA Component 1": -0.5742343122961749, "PCA Component 2": 0.7781306765729349}, {"Overall rank": 135, "Country or region": "Eswatini", "Score": 4.212, "GDP per capita": 0.8109999999999999, "Social support": 1.149, "Healthy life expectancy": 0.0, "Freedom to make life choices": 0.313, "Generosity": 0.07400000000000001, "Perceptions of corruption": 0.135, "PCA Component 1": -0.9318114565073825, "PCA Component 2": 0.427156740439652}, {"Overall rank": 136, "Country or region": "Uganda", "Score": 4.189, "GDP per capita": 0.332, "Social support": 1.069, "Healthy life expectancy": 0.44299999999999995, "Freedom to make life choices": 0.35600000000000004, "Generosity": 0.252, "Perceptions of corruption": 0.06, "PCA Component 1": -0.5949546530865197, "PCA Component 2": 0.7917530948939675}, {"Overall rank": 137, "Country or region": "Egypt", "Score": 4.166, "GDP per capita": 0.9129999999999999, "Social support": 1.0390000000000001, "Healthy life expectancy": 0.644, "Freedom to make life choices": 0.24100000000000002, "Generosity": 0.076, "Perceptions of corruption": 0.067, "PCA Component 1": -0.8466073080951843, "PCA Component 2": 0.4675981469992577}, {"Overall rank": 138, "Country or region": "Zambia", "Score": 4.107, "GDP per capita": 0.578, "Social support": 1.058, "Healthy life expectancy": 0.426, "Freedom to make life choices": 0.431, "Generosity": 0.247, "Perceptions of corruption": 0.087, "PCA Component 1": -0.7367930442794627, "PCA Component 2": 0.6357176751422461}, {"Overall rank": 139, "Country or region": "Togo", "Score": 4.085, "GDP per capita": 0.275, "Social support": 0.5720000000000001, "Healthy life expectancy": 0.41, "Freedom to make life choices": 0.293, "Generosity": 0.177, "Perceptions of corruption": 0.085, "PCA Component 1": -0.36615732806225976, "PCA Component 2": 0.4204897539332644}, {"Overall rank": 140, "Country or region": "India", "Score": 4.015, "GDP per capita": 0.755, "Social support": 0.765, "Healthy life expectancy": 0.588, "Freedom to make life choices": 0.498, "Generosity": 0.2, "Perceptions of corruption": 0.085, "PCA Component 1": -0.7198672297320096, "PCA Component 2": 0.3426102935410412}, {"Overall rank": 141, "Country or region": "Liberia", "Score": 3.975, "GDP per capita": 0.073, "Social support": 0.922, "Healthy life expectancy": 0.44299999999999995, "Freedom to make life choices": 0.37, "Generosity": 0.233, "Perceptions of corruption": 0.033, "PCA Component 1": -0.4185830650891308, "PCA Component 2": 0.8190331351905369}, {"Overall rank": 142, "Country or region": "Comoros", "Score": 3.9730000000000003, "GDP per capita": 0.27399999999999997, "Social support": 0.757, "Healthy life expectancy": 0.505, "Freedom to make life choices": 0.142, "Generosity": 0.275, "Perceptions of corruption": 0.078, "PCA Component 1": -0.3615193021507009, "PCA Component 2": 0.6185280519129741}, {"Overall rank": 143, "Country or region": "Madagascar", "Score": 3.9330000000000003, "GDP per capita": 0.27399999999999997, "Social support": 0.9159999999999999, "Healthy life expectancy": 0.555, "Freedom to make life choices": 0.14800000000000002, "Generosity": 0.16899999999999998, "Perceptions of corruption": 0.040999999999999995, "PCA Component 1": -0.44542530184179985, "PCA Component 2": 0.7313630546809806}, {"Overall rank": 144, "Country or region": "Lesotho", "Score": 3.802, "GDP per capita": 0.489, "Social support": 1.169, "Healthy life expectancy": 0.168, "Freedom to make life choices": 0.359, "Generosity": 0.107, "Perceptions of corruption": 0.09300000000000001, "PCA Component 1": -0.7720529850483494, "PCA Component 2": 0.6736639804745984}, {"Overall rank": 145, "Country or region": "Burundi", "Score": 3.775, "GDP per capita": 0.046, "Social support": 0.447, "Healthy life expectancy": 0.38, "Freedom to make life choices": 0.22, "Generosity": 0.17600000000000002, "Perceptions of corruption": 0.18, "PCA Component 1": -0.17636629562511538, "PCA Component 2": 0.4487162573829138}, {"Overall rank": 146, "Country or region": "Zimbabwe", "Score": 3.6630000000000003, "GDP per capita": 0.366, "Social support": 1.114, "Healthy life expectancy": 0.433, "Freedom to make life choices": 0.361, "Generosity": 0.151, "Perceptions of corruption": 0.08900000000000001, "PCA Component 1": -0.6507438614367755, "PCA Component 2": 0.7828223847007515}, {"Overall rank": 147, "Country or region": "Haiti", "Score": 3.597, "GDP per capita": 0.32299999999999995, "Social support": 0.688, "Healthy life expectancy": 0.449, "Freedom to make life choices": 0.026000000000000002, "Generosity": 0.419, "Perceptions of corruption": 0.11, "PCA Component 1": -0.2964708655987447, "PCA Component 2": 0.5598203689432476}, {"Overall rank": 148, "Country or region": "Botswana", "Score": 3.488, "GDP per capita": 1.041, "Social support": 1.145, "Healthy life expectancy": 0.5379999999999999, "Freedom to make life choices": 0.455, "Generosity": 0.025, "Perceptions of corruption": 0.1, "PCA Component 1": -1.03946699063037, "PCA Component 2": 0.42550634304310836}, {"Overall rank": 149, "Country or region": "Syria", "Score": 3.4619999999999997, "GDP per capita": 0.619, "Social support": 0.37799999999999995, "Healthy life expectancy": 0.44, "Freedom to make life choices": 0.013000000000000001, "Generosity": 0.331, "Perceptions of corruption": 0.141, "PCA Component 1": -0.33200321139266886, "PCA Component 2": 0.13553431758544313}, {"Overall rank": 150, "Country or region": "Malawi", "Score": 3.41, "GDP per capita": 0.191, "Social support": 0.56, "Healthy life expectancy": 0.495, "Freedom to make life choices": 0.44299999999999995, "Generosity": 0.218, "Perceptions of corruption": 0.08900000000000001, "PCA Component 1": -0.3485049628252317, "PCA Component 2": 0.48233736033792945}, {"Overall rank": 151, "Country or region": "Yemen", "Score": 3.38, "GDP per capita": 0.287, "Social support": 1.163, "Healthy life expectancy": 0.46299999999999997, "Freedom to make life choices": 0.14300000000000002, "Generosity": 0.10800000000000001, "Perceptions of corruption": 0.077, "PCA Component 1": -0.5688652507528258, "PCA Component 2": 0.8757649973154153}, {"Overall rank": 152, "Country or region": "Rwanda", "Score": 3.3339999999999996, "GDP per capita": 0.359, "Social support": 0.711, "Healthy life expectancy": 0.614, "Freedom to make life choices": 0.555, "Generosity": 0.217, "Perceptions of corruption": 0.41100000000000003, "PCA Component 1": -0.48850602748889693, "PCA Component 2": 0.5320951471241148}, {"Overall rank": 153, "Country or region": "Tanzania", "Score": 3.2310000000000003, "GDP per capita": 0.47600000000000003, "Social support": 0.885, "Healthy life expectancy": 0.499, "Freedom to make life choices": 0.41700000000000004, "Generosity": 0.276, "Perceptions of corruption": 0.147, "PCA Component 1": -0.594046981515744, "PCA Component 2": 0.5873958991632076}, {"Overall rank": 154, "Country or region": "Afghanistan", "Score": 3.2030000000000003, "GDP per capita": 0.35, "Social support": 0.517, "Healthy life expectancy": 0.361, "Freedom to make life choices": 0.0, "Generosity": 0.158, "Perceptions of corruption": 0.025, "PCA Component 1": -0.3034489934775598, "PCA Component 2": 0.3344859618796185}, {"Overall rank": 155, "Country or region": "Central African Republic", "Score": 3.083, "GDP per capita": 0.026000000000000002, "Social support": 0.0, "Healthy life expectancy": 0.105, "Freedom to make life choices": 0.225, "Generosity": 0.235, "Perceptions of corruption": 0.035, "PCA Component 1": -0.020931557227160006, "PCA Component 2": 0.05232595091786317}, {"Overall rank": 156, "Country or region": "South Sudan", "Score": 2.853, "GDP per capita": 0.306, "Social support": 0.575, "Healthy life expectancy": 0.295, "Freedom to make life choices": 0.01, "Generosity": 0.20199999999999999, "Perceptions of corruption": 0.091, "PCA Component 1": -0.30142241051324226, "PCA Component 2": 0.3945538669136417}]}}, {"mode": "vega-lite"});
</script>



#### Eigenfaces

A common recent application of PCA is the so-called 'eigenfaces'. That is, if we apply PCA to a dataset of images of faces we can attempt to generate a compact code for faces. For example, suppose we use 100x100 face images, this means that we effectively have a 10000-dimensional space in which all faces must exist. Obviously this is rather unwieldy and so, say, if we could generate say a code of 400 real numbers for a face, it could be useful for compressive purposes or indeed for face recognition.

_Eigenfaces_ specifically refers to the projection vectors which applying PCA to faces generates. These essentially represent the directions of most variance in our original 10000-dimensional space and help to point out some interesting things that PCA is doing.

To test our implementation out on faces, I used the following Kaggle [dataset](https://www.kaggle.com/abhikjha/utk-face-cropped), which you can download to path in the code to reproduce the notebook.

Interestingly we can take a look at the 'mean' (in the sense of average) face --- pretty generic.


```python
imshow(Image.fromarray(np.uint8(np.mean(imag_arr, axis=0)).reshape(reversed_dims)), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x13d773c10>




![png](pca_files/pca_51_1.png)


We can use PCA to reduce the number of components of our image data from 2400 to some new reduced number of dimensions. Given this new low-dimensional representation we'll try to reconstruct the original and take a look at the reconstructed image.
As we can see, people frequently become recognisable well before 2400 dimensions. Some people are even recognisable with as little as a few hundred components.

If you rerun the cell, it will pick a person at random from the dataset and show this grid of images again.


```python
def reconstruct(num_components, encoding_matrix, test_image):
    proj = encoding_matrix[:,:num_components]
    rec = (test_image@proj@proj.T).reshape(reversed_dims)
    return Image.fromarray(np.uint8(rec))
    
```


![png](pca_files/pca_55_0.png)


PCA, in some sense, is quite interpretable. We can take a look at exactly what it's considering, that is, the directions of the eigenvectors it's using.

In this specific case, we can look at and interpret them as images which each test image is being projected onto.


![png](pca_files/pca_57_0.png)


If you thought that looked weird, let's look at some of the other components. They get more and more bizarre at a first glance. However, we know that most faces roughly look the same, so later eigenvectors will be distinguishing less and less, hence why they become less obvious.


![png](pca_files/pca_59_0.png)


Thanks for reading I hope it was useful or at least mildly interesting.

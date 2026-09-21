# autograd

[![PkgGoDev](https://pkg.go.dev/badge/github.com/itsubaki/autograd)](https://pkg.go.dev/github.com/itsubaki/autograd)
[![tests](https://github.com/itsubaki/autograd/workflows/tests/badge.svg)](https://github.com/itsubaki/autograd/actions)

An automatic differentiation library in Go.

## Examples

### Backward

```go
import (
	"fmt"
	
	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func main() {
	x := variable.New(1.0)
	y := F.Sin(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
}

// Output:
// variable(0.84147096)
// variable(0.5403023)
```

### Composite function

```go
matyas := func(x, y *variable.Variable) *variable.Variable {
	// 0.26(x^2 + y^2) - 0.48xy
	z0 := F.MulC(0.26, F.Add(F.Pow(2.0)(x), F.Pow(2.0)(y)))
	z1 := F.MulC(0.48, F.Mul(x, y))
	return F.Sub(z0, z1)
}

x := variable.New(1.0)
y := variable.New(1.0)
z := matyas(x, y)
z.Backward()

fmt.Println(x.Grad)
fmt.Println(y.Grad)

// Output:
// variable(0.03999999)
// variable(0.03999999)
```

### Gradient descent

```go
rosenbrock := func(x0, x1 *variable.Variable) *variable.Variable {
	// 100 * (x1 - x0^2)^2 + (x0 - 1)^2
	y0 := F.Pow(2.0)(F.Sub(x1, F.Pow(2.0)(x0)))
	y1 := F.Pow(2.0)(F.AddC(-1.0, x0))
	return F.Add(F.MulC(100, y0), y1)
}

update := func(lr float64, x ...*variable.Variable) {
	for _, v := range x {
		v.Data = tensor.F2(v.Data, v.Grad.Data, func(a, b float64) float64 {
			return a - lr*b
		})
	}
}

x0 := variable.New(0.0)
x1 := variable.New(2.0)

lr := 0.001
iters := 10000

for i := range iters + 1 {
	if i%1000 == 0 {
		fmt.Println(x0, x1)
	}

	x0.Cleargrad()
	x1.Cleargrad()
	y := rosenbrock(x0, x1)
	y.Backward()

	update(lr, x0, x1)
}

// Output:
// variable(0)         variable(2)
// variable(0.6837119) variable(0.4659528)
// variable(0.8263181) variable(0.6820318)
// variable(0.8947841) variable(0.8001903)
// variable(0.9334872) variable(0.8711214)
// variable(0.9569893) variable(0.9156519)
// variable(0.9718162) variable(0.9443121)
// variable(0.9813804) variable(0.9630323)
// variable(0.9876351) variable(0.9753732)
// variable(0.9917611) variable(0.9835568)
// variable(0.9944981) variable(0.9890044)
```

### Deep Learning

```go
dataset := NewCurve(N, noise, math.Sin)
dataloader := &DataLoader{
	BatchSize: batchSize,
	N:         dataset.N,
	Data:      dataset.Data,
	Label:     dataset.Label,
}

m := model.NewLSTM(hiddenSize, 1)
o := optimizer.SGD{
	LearningRate: 0.01,
}

for i := range epochs {
	m.ResetState()

	loss, count := variable.New(0), 0
	for x, t := dataloader.Seq2() {
		y := m.Forward(x)
		loss = F.Add(loss, F.MeanSquaredError(y, t))

		if count++; count%bpttLength == 0 || count == dataset.N {
			m.Cleargrads()
			loss.Backward()
			loss.UnchainBackward()
			o.Update(m)
		}
	}
}
```

### Double backpropagation

```go
x := variable.New(1.0)
y := F.Sin(x)
y.Backward(variable.Opts{
	CreateGraph: true,
})

fmt.Println(y)
fmt.Println(x.Grad)

for range 5 {
	gx := x.Grad
	x.Cleargrad()
	gx.Backward(variable.Opts{
		CreateGraph: true,
	})

	fmt.Println(x.Grad)
}

// Output:
// variable(0.84147096)
// variable(0.5403023)
// variable(-0.84147096)
// variable(-0.5403023)
// variable(0.84147096)
// variable(0.5403023)
// variable(-0.84147096)
```

### NoGrad and Test mode

```go
func() {
	defer variable.Nograd().End()

	// No graphs are generated for gradient computation.
	for _, x := range xs {
		m.Forward(x)
	}
}()
```

```go
func() {
	defer variable.TestMode().End()

	// DropoutSimple has no effect during test mode.
	// Take a look at the implementation of DropoutSimple as well.
	F.DropoutSimple(0.5)(x)
}()
```

### Dot graph

```shell
brew install graphviz
```

```shell
go run cmd/dot/main.go -func tanh -order 2 -verbose > sample.dot
dot sample.dot -T png -o sample.png
```

<img src="https://github.com/itsubaki/autograd/blob/main/dtanh.png" height="240px"><img src="https://github.com/itsubaki/autograd/blob/main/dtanh2.png" height="240px"><img src="https://github.com/itsubaki/autograd/blob/main/dtanh3.png" height="240px">

## References

- [ゼロから作るDeep Learning ❸](https://www.oreilly.co.jp/books/9784873119069/)
- [oreilly-japan/deep-learning-from-scratch-3](https://github.com/oreilly-japan/deep-learning-from-scratch-3)

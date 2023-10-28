# autograd

[![PkgGoDev](https://pkg.go.dev/badge/github.com/itsubaki/autograd)](https://pkg.go.dev/github.com/itsubaki/autograd)
[![Go Report Card](https://goreportcard.com/badge/github.com/itsubaki/autograd?style=flat-square)](https://goreportcard.com/report/github.com/itsubaki/autograd)
[![tests](https://github.com/itsubaki/autograd/workflows/tests/badge.svg?branch=main)](https://github.com/itsubaki/autograd/actions)
[![codecov](https://codecov.io/gh/itsubaki/autograd/graph/badge.svg?token=loXkcn2w9W)](https://codecov.io/gh/itsubaki/autograd)

- Automatic differentiation library for Go
- pure Go implementation
- using only the standard library

# Example

## Composite functions

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

fmt.Println(x.Grad, y.Grad)

// Output:
// variable([0.040000000000000036]) variable([0.040000000000000036])
```

## Gradient descent

```go
rosenbrock := func(x0, x1 *variable.Variable) *variable.Variable {
	// 100 * (x1 - x0^2)^2 + (x0 - 1)^2
	y0 := F.Pow(2.0)(F.Sub(x1, F.Pow(2.0)(x0)))
	y1 := F.Pow(2.0)(F.AddC(-1.0, x0))
	return F.Add(F.MulC(100, y0), y1)
}

update := func(lr float64, x ...*variable.Variable) {
	for _, v := range x {
		v.Data = vector.F2(v.Data, v.Grad.Data, func(a, b float64) float64 {
			return a - lr*b
		})
	}
}

x0 := variable.New(0.0)
x1 := variable.New(2.0)

lr := 0.001
iters := 10000

for i := 0; i < iters+1; i++ {
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
// variable([0]) variable([2])
// variable([0.6837118569138317]) variable([0.4659526837427042])
// variable([0.8263177857050957]) variable([0.6820311873361097])
// variable([0.8947837494333546]) variable([0.8001896451930564])
// variable([0.9334871723401226]) variable([0.8711213202579401])
// variable([0.9569899983530249]) variable([0.9156532462021957])
// variable([0.9718168065095137]) variable([0.9443132014542008])
// variable([0.9813809710644894]) variable([0.9630332658658076])
// variable([0.9876355102559093]) variable([0.9753740541653942])
// variable([0.9917613994572028]) variable([0.9835575421346807])
// variable([0.9944984367782456]) variable([0.9890050527419593])
```

## Deep Learning

```go
dataset := NewCurve(math.Sin)
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

for i := 0; i < epoch; i++ {
	m.ResetState()

	loss, count := variable.New(0), 0
	for dataloader.Next() {
		x, t := dataloader.Batch()
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

## Double backpropagation

```go
x := variable.New(1.0)
y := F.Sin(x)
y.Backward(variable.Opts{CreateGraph: true})

fmt.Println(y)
fmt.Println(x.Grad)

for i := 0; i < 5; i++ {
	gx := x.Grad
	x.Cleargrad()
	gx.Backward(variable.Opts{CreateGraph: true})

	fmt.Println(x.Grad)
}

// Output:
// variable([0.8414709848078965])
// variable([0.5403023058681398])
// variable([-0.8414709848078965])
// variable([-0.5403023058681398])
// variable([0.8414709848078965])
// variable([0.5403023058681398])
// variable([-0.8414709848078965])
```

## Dot graph

```shell
$ brew install graphviz
```

```shell
$ go run cmd/dot/main.go -func tanh -order 2 -verbose > sample.dot
$ dot sample.dot -T png -o sample.png
```

<img src="https://github.com/itsubaki/autograd/blob/main/dtanh.png" height="240px"><img src="https://github.com/itsubaki/autograd/blob/main/dtanh2.png" height="240px"><img src="https://github.com/itsubaki/autograd/blob/main/dtanh3.png" height="240px">

# Links

- [oreilly-japan/deep-learning-from-scratch-3](https://github.com/oreilly-japan/deep-learning-from-scratch-3)
- [oreilly-japan/deep-learning-from-scratch-3/tree/tanh](https://github.com/oreilly-japan/deep-learning-from-scratch-3/tree/tanh)

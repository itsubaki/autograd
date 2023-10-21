package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

type SinCurve struct {
	N         int
	BatchSize int
	Data      []float64
	Label     []float64
	iter      int
}

func NewSinCurve(batchSize int) *SinCurve {
	N, noise := 1000, 0.05

	x := make([]float64, N)
	for i := 0; i < N; i++ {
		x[i] = float64(i) * 2 * math.Pi / float64(N-1)
	}

	y := make([]float64, N)
	for i := 0; i < N; i++ {
		y[i] = math.Sin(x[i]) + rand.Float64()*(2*noise) - noise // noise
	}

	return &SinCurve{
		N:         N,
		BatchSize: batchSize,
		Data:      y[:len(x)-1],
		Label:     y[1:],
		iter:      0,
	}
}

func (d *SinCurve) Next() bool {
	next := (d.iter+1)*d.BatchSize < d.N
	if !next {
		d.iter = 0
	}

	return next
}

func (d *SinCurve) Read() (*variable.Variable, *variable.Variable) {
	begin, end := d.iter*d.BatchSize, (d.iter+1)*d.BatchSize
	x, y := vector.Transpose(d.Data[begin:end]), vector.Transpose(d.Label[begin:end])
	d.iter++
	return variable.NewOf(x...), variable.NewOf(y...)
}

func main() {
	var epoch, batchSize, hiddenSize, bpttLength int
	flag.IntVar(&epoch, "epoch", 100, "")
	flag.IntVar(&batchSize, "batch-size", 30, "")
	flag.IntVar(&hiddenSize, "hidden-size", 100, "")
	flag.IntVar(&bpttLength, "bptt-length", 30, "")
	flag.Parse()

	m := model.NewLSTM(hiddenSize, 1)
	o := optimizer.SGD{LearningRate: 0.01}

	dataset := NewSinCurve(batchSize)
	for i := 0; i < epoch; i++ {
		m.ResetState()

		loss, count := variable.Const(0), 0
		for dataset.Next() {
			x, t := dataset.Read()
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

	// cos curve
	xs := make([]float64, dataset.N)
	for i := 0; i < len(xs); i++ {
		xs[i] = math.Cos(float64(i) * 4 * math.Pi / float64(len(xs)-1))
	}

	// predict
	ys := make([]float64, len(xs))
	func() {
		defer variable.Nograd().End()

		m.ResetState()
		for i, x := range xs {
			y := m.Forward(variable.New(x))
			ys[i] = y.Data[0][0]
		}
	}()

	x := make([]float64, len(ys))
	for i := 0; i < len(x); i++ {
		x[i] = float64(i)
	}

	// to csv
	for i := 0; i < len(x); i++ {
		fmt.Printf("%f,%f\n", x[i], xs[i])
	}
}
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

type Sequence struct {
	N     int
	Data  []float64
	Label []float64
}

func NewCurve(f func(x float64) float64) *Sequence {
	N, noise := 1000, 0.05

	y := make([]float64, N)
	for i := 0; i < N; i++ {
		x := 2 * math.Pi * float64(i) / float64(N-1)
		y[i] = f(x) + rand.Float64()*(2*noise) - noise
	}

	return &Sequence{
		N:     N,
		Data:  y[:len(y)-1],
		Label: y[1:],
	}
}

type DataLoader struct {
	BatchSize int
	N         int
	Data      []float64
	Label     []float64
	iter      int
}

func (d *DataLoader) Next() bool {
	next := (d.iter+1)*d.BatchSize < d.N
	if !next {
		d.iter = 0
	}

	return next
}

func (d *DataLoader) Batch() (*variable.Variable, *variable.Variable) {
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

	dataset := NewCurve(math.Sin)
	dataloader := &DataLoader{
		BatchSize: batchSize,
		N:         dataset.N,
		Data:      dataset.Data,
		Label:     dataset.Label,
	}

	m := model.NewLSTM(hiddenSize, 1)
	o := optimizer.SGD{LearningRate: 0.01}

	for i := 0; i < epoch; i++ {
		m.ResetState()

		loss, count := variable.Const(0), 0
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

	// cos curve
	xs := make([]float64, dataset.N)
	for i := 0; i < len(xs); i++ {
		xs[i] = math.Cos(4 * math.Pi * float64(i) / float64(len(xs)-1))
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

	// csv
	x := make([]float64, len(ys))
	for i := 0; i < len(x); i++ {
		x[i] = float64(i)
	}

	for i := 0; i < len(x); i++ {
		fmt.Printf("%f,%f\n", x[i], ys[i])
	}
}

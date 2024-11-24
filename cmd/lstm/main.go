package main

import (
	"flag"
	"fmt"
	"iter"
	"log"
	"math"
	"math/rand/v2"
	"time"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

// pi returns a slice of float64 in [0, c*PI].
func pi(c float64, N int) []float64 {
	xs := make([]float64, N)
	for i := 0; i < N; i++ {
		xs[i] = c * math.Pi * (float64(i) / float64(N-1))
	}

	return xs
}

type Sequence struct {
	N     int
	Data  []float64
	Label []float64
}

func NewCurve(N int, noise float64, f func(x float64) float64) *Sequence {
	y := make([]float64, N)
	for i, x := range pi(2, N) {
		y[i] = f(x) + rand.Float64()*(2*noise) - noise
	}

	return &Sequence{
		N:     N - 1,
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

func (l *DataLoader) Next() bool {
	next := (l.iter+1)*l.BatchSize < l.N
	if !next {
		l.iter = 0
	}

	return next
}

func (l *DataLoader) Batch() (*variable.Variable, *variable.Variable) {
	begin, end := l.iter*l.BatchSize, (l.iter+1)*l.BatchSize
	x, y := vector.Transpose(l.Data[begin:end]), vector.Transpose(l.Label[begin:end])
	l.iter++
	return variable.NewOf(x...), variable.NewOf(y...)
}

func (l *DataLoader) Seq2() iter.Seq2[*variable.Variable, *variable.Variable] {
	return func(yield func(*variable.Variable, *variable.Variable) bool) {
		for l.Next() {
			x, t := l.Batch()
			if !yield(x, t) {
				break
			}
		}
	}
}

func main() {
	var N, epochs, batchSize, hiddenSize, bpttLength int
	var noise, lr float64
	flag.IntVar(&N, "N", 1000, "")
	flag.IntVar(&epochs, "epochs", 100, "")
	flag.IntVar(&batchSize, "batch-size", 30, "")
	flag.IntVar(&hiddenSize, "hidden-size", 100, "")
	flag.IntVar(&bpttLength, "bptt-length", 30, "")
	flag.Float64Var(&lr, "learning-rate", 0.01, "")
	flag.Float64Var(&noise, "noise", 0.05, "")
	flag.Parse()

	dataset := NewCurve(N, noise, math.Sin)
	dataloader := &DataLoader{
		BatchSize: batchSize,
		N:         dataset.N,
		Data:      dataset.Data,
		Label:     dataset.Label,
	}

	m := model.NewLSTM(hiddenSize, 1)
	o := optimizer.SGD{
		LearningRate: lr,
	}

	now := time.Now()
	for i := 0; i < epochs; i++ {
		m.ResetState()

		loss, count := variable.New(0), 0
		for x, t := range dataloader.Seq2() {
			y := m.Forward(x)
			loss = F.Add(loss, F.MeanSquaredError(y, t))

			if count++; count%bpttLength == 0 || count == dataset.N {
				m.Cleargrads()
				loss.Backward()
				loss.UnchainBackward()
				o.Update(m)
			}
		}

		log.Printf("%3d: %f\n", i, loss.Data[0][0]/float64(count))
	}
	log.Printf("elapsed=%v\n", time.Since(now))

	// cos curve
	xs := make([]float64, dataset.N)
	for i, x := range pi(4, len(xs)) {
		xs[i] = math.Cos(x)
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
	for i, x := range pi(4, len(xs)) {
		fmt.Printf("%f,%f\n", x, ys[i])
	}
}

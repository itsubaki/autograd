package tensor_test

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/tensor"
)

func Example() {
	v := tensor.Zero[float64](2, 3)

	v.Set([]int{0, 0}, 1.5)
	v.Set([]int{0, 1}, 2.5)
	v.Set([]int{0, 2}, 3.5)
	v.Set([]int{1, 0}, 4.5)
	v.Set([]int{1, 1}, 5.5)
	v.Set([]int{1, 2}, 6.5)

	fmt.Println(v.At(0, 0), v.At(0, 1), v.At(0, 2))
	fmt.Println(v.At(1, 0), v.At(1, 1), v.At(1, 2))

	// Output:
	// 1.5 2.5 3.5
	// 4.5 5.5 6.5
}

func ExampleFull() {
	v := tensor.Full([]int{2, 3}, 3.14)
	fmt.Println(v.Shape)
	fmt.Println(v.Data)

	// Output:
	// [2 3]
	// [3.14 3.14 3.14 3.14 3.14 3.14]
}

func ExampleZeroLike() {
	v := tensor.Zero[int](2, 3)
	w := tensor.ZeroLike(v)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 3]
	// [0 0 0 0 0 0]
}

func ExampleOneLike() {
	v := tensor.Zero[int](2, 3)
	w := tensor.OneLike(v)
	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 3]
	// [1 1 1 1 1 1]
}

func ExampleRand() {
	v := tensor.Rand([]int{2, 3})
	fmt.Println(v.Shape)

	// Output:
	// [2 3]
}

func ExampleRand_seed() {
	s := rand.Const()
	v := tensor.Rand([]int{2, 3}, s)

	fmt.Printf("%.4f, %.4f, %.4f\n", v.At(0, 0), v.At(0, 1), v.At(0, 2))
	fmt.Printf("%.4f, %.4f, %.4f\n", v.At(1, 0), v.At(1, 1), v.At(1, 2))

	// Output:
	// 0.9999, 0.8856, 0.3815
	// 0.4813, 0.4442, 0.5210
}

func ExampleRand_nil() {
	v := tensor.Rand([]int{2, 3}, nil)
	fmt.Println(v.Shape)

	// Output:
	// [2 3]
}

func ExampleRandn() {
	v := tensor.Randn([]int{2, 3})
	fmt.Println(v.Shape)

	// Output:
	// [2 3]
}

func ExampleRandn_seed() {
	s := rand.Const()
	v := tensor.Randn([]int{2, 3}, s)

	fmt.Printf("%.4f, %.4f, %.4f\n", v.At(0, 0), v.At(0, 1), v.At(0, 2))
	fmt.Printf("%.4f, %.4f, %.4f\n", v.At(1, 0), v.At(1, 1), v.At(1, 2))

	// Output:
	// 0.5665, -0.6124, 0.5899
	// -0.3678, 1.0920, -0.4438
}

func ExampleReshape() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.Reshape(v, 1, 4)

	fmt.Println(w.Shape)
	fmt.Println(w.At(0, 0), w.At(0, 1), w.At(0, 2), w.At(0, 3))

	// Output:
	// [1 4]
	// 1 2 3 4
}

func ExampleReshape_invalid() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println(r)
			return
		}

		panic("unexpected panic for index")
	}()

	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	_ = tensor.Reshape(v, 10, 10)

	panic("unreachable")

	// Output:
	// invalid shape
}

func ExampleTensor_Clone() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := v.Clone()
	w.Set([]int{0, 0}, 10)

	fmt.Println(v.Data)
	fmt.Println(w.Data)

	// Output:
	// [1 2 3 4]
	// [10 2 3 4]
}

func ExampleTensor_Float64() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := v.Float64()

	fmt.Printf("%T", w.Data)

	// Output:
	// []float64
}

func ExampleTensor_AddAt() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	v.AddAt([]int{0, 0}, 10)

	fmt.Println(v.Data)

	// Output:
	// [11 2 3 4]
}

func ExampleTensor_ScatterAdd() {
	v := tensor.New([]int{3, 2}, []int{
		10, 11,
		20, 21,
		30, 31,
	})
	w := tensor.New([]int{2, 2}, []int{
		1, 2,
		3, 4,
	})

	v.ScatterAdd(w, []int{0, 2}, 0)

	fmt.Println(v.Shape)
	fmt.Println(v.Data)

	// Output:
	// [3 2]
	// [11 13 20 21 33 35]
}

func ExampleTensor_ScatterAdd_axis1() {
	v := tensor.New([]int{3, 2}, []int{
		10, 11,
		20, 21,
		30, 31,
	})
	w := tensor.New([]int{3, 2}, []int{
		1, 2,
		2, 3,
		3, 6,
	})

	v.ScatterAdd(w, []int{0, 0}, 1)

	fmt.Println(v.Shape)
	fmt.Println(v.Data)

	// Output:
	// [3 2]
	// [13 11 25 21 39 31]
}

func ExampleTake() {
	v := tensor.New([]int{3, 2}, []int{
		10, 11,
		20, 21,
		30, 31,
	})

	w := tensor.Take(v, []int{0, 2}, 0)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 2]
	// [10 11 30 31]
}

func ExampleAddC() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.AddC(10, v)

	fmt.Println(w.Data)

	// Output:
	// [11 12 13 14]
}

func ExampleSubC() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.SubC(10, v)

	fmt.Println(w.Data)

	// Output:
	// [9 8 7 6]
}

func ExampleMulC() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.MulC(10, v)

	fmt.Println(w.Data)

	// Output:
	// [10 20 30 40]
}

func ExamplePow() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Pow(3, v)

	fmt.Println(w.Data)

	// Output:
	// [1 8 27 64]
}

func ExampleExp() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Exp(v)

	fmt.Printf("%.4f\n", w.Data)

	// Output:
	// [2.7183 7.3891 20.0855 54.5982]
}

func ExampleLog() {
	data := []float64{math.Exp(0), math.Exp(1), math.Exp(2), math.Exp(3)}
	v := tensor.New([]int{2, 2}, data)
	w := tensor.Log(v)

	fmt.Printf("%.4f\n", w.Data)

	// Output:
	// [0.0000 1.0000 2.0000 3.0000]
}

func ExampleSin() {
	data := []float64{0, math.Pi / 2, math.Pi, 3 * math.Pi / 2}
	v := tensor.New([]int{2, 2}, data)
	w := tensor.Sin(v)

	fmt.Printf("%.4f\n", w.Data)

	// Output:
	// [0.0000 1.0000 0.0000 -1.0000]
}

func ExampleCos() {
	data := []float64{0, math.Pi / 2, math.Pi, 3 * math.Pi / 2}
	v := tensor.New([]int{2, 2}, data)
	w := tensor.Cos(v)

	fmt.Printf("%.4f\n", w.Data)

	// Output:
	// [1.0000 0.0000 -1.0000 -0.0000]
}

func ExampleTanh() {
	v := tensor.New([]int{2, 2}, []float64{-1, 0, 1, 2})
	w := tensor.Tanh(v)

	fmt.Printf("%.4f\n", w.Data)

	// Output:
	// [-0.7616 0.0000 0.7616 0.9640]
}

func ExampleAdd() {
	x := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	y := tensor.New([]int{2, 2}, []int{10, 20, 30, 40})
	z := tensor.Add(x, y)

	fmt.Println(z.Data)

	// Output:
	// [11 22 33 44]
}

func ExampleAdd_broadcast0() {
	x := tensor.New([]int{1, 2}, []int{1, 2})
	y := tensor.New([]int{2, 2}, []int{10, 20, 30, 40})
	z := tensor.Add(x, y)

	fmt.Println(z.Shape)
	fmt.Println(z.Data)

	// Output:
	// [2 2]
	// [11 22 31 42]
}

func ExampleAdd_broadcast1() {
	x := tensor.New([]int{2, 1}, []int{1, 2})
	y := tensor.New([]int{2, 2}, []int{10, 20, 30, 40})
	z := tensor.Add(x, y)

	fmt.Println(z.Shape)
	fmt.Println(z.Data)

	// Output:
	// [2 2]
	// [11 21 32 42]
}

func ExampleSub() {
	x := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	y := tensor.New([]int{2, 2}, []int{10, 20, 30, 40})
	z := tensor.Sub(x, y)

	fmt.Println(z.Data)

	// Output:
	// [-9 -18 -27 -36]
}

func ExampleMul() {
	x := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	y := tensor.New([]int{2, 2}, []int{10, 20, 30, 40})
	z := tensor.Mul(x, y)

	fmt.Println(z.Data)

	// Output:
	// [10 40 90 160]
}

func ExampleDiv() {
	x := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	y := tensor.New([]int{2, 2}, []float64{10, 20, 30, 40})
	z := tensor.Div(x, y)

	fmt.Printf("%.4f\n", z.Data)

	// Output:
	// [0.1000 0.1000 0.1000 0.1000]
}

func ExampleSum() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.Sum(v)

	fmt.Println(w.At())

	// Output:
	// 10
}

func ExampleSum_axisAll() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.Sum(v, 0, 1)

	fmt.Println(w.Data)

	// Output:
	// [10]
}

func ExampleSum_axis0() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.Sum(v, 0)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [4 6]
}

func ExampleSum_axis1() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.Sum(v, 1)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [3 7]
}

func ExampleMax() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.Max(v)

	fmt.Println(w.At())

	// Output:
	// 4
}

func ExampleMax_axisAll() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.Max(v, 0, 1)

	fmt.Println(w.At())

	// Output:
	// 4
}

func ExampleMax_axis0() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.Max(v, 0)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [3 4]
}

func ExampleMax_axis1() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.Max(v, 1)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [2 4]
}

func ExampleMin() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.Min(v)

	fmt.Println(w.At())

	// Output:
	// 1
}

func ExampleMin_axisAll() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.Min(v, 0, 1)

	fmt.Println(w.At())

	// Output:
	// 1
}

func ExampleMin_axis0() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.Min(v, 0)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [1 2]
}

func ExampleMin_axis1() {
	v := tensor.New([]int{2, 2}, []int{1, 2, 3, 4})
	w := tensor.Min(v, 1)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [1 3]
}

func ExampleMean() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Mean(v)

	fmt.Println(w.At())

	// Output:
	// 2.5
}

func ExampleMean_axisAll() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Mean(v, 0, 1)

	fmt.Println(w.At())

	// Output:
	// 2.5
}

func ExampleMean_axis0() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Mean(v, 0)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [2 3]
}

func ExampleMean_axis1() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Mean(v, 1)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [1.5 3.5]
}

func ExampleArgmax_axis0() {
	v := tensor.New([]int{2, 4}, []int{
		1, 2, 3, 4,
		4, 3, 2, 1,
	})
	w := tensor.Argmax(v, 0)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [4]
	// [1 1 0 0]
}

func ExampleArgmax_axis1() {
	v := tensor.New([]int{2, 4}, []int{
		1, 2, 3, 4,
		4, 3, 2, 1,
	})
	w := tensor.Argmax(v, 1)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [3 0]
}

func ExampleMask() {
	v := tensor.New([]int{2, 2}, []int{-1, 2, -3, 4})
	w := tensor.Mask(v, func(v int) bool { return v > 0 })

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 2]
	// [0 1 0 1]
}

func ExampleClip() {
	v := tensor.New([]int{2, 10}, []int{
		-3, -2, -1, 0,
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		11, 12, 13, 14, 15, 16,
	})
	w := tensor.Clip(v, 0, 10)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 10]
	// [0 0 0 0 1 2 3 4 5 6 7 8 9 10 10 10 10 10 10 10]
}

func ExampleTranspose() {
	v := tensor.New([]int{2, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Transpose(v)

	fmt.Println(w.Shape)
	fmt.Println(w.At(0, 0), w.At(0, 1))
	fmt.Println(w.At(1, 0), w.At(1, 1))
	fmt.Println(w.At(2, 0), w.At(2, 1))

	// Output:
	// [3 2]
	// 1 4
	// 2 5
	// 3 6
}

func ExampleTranspose_axes01() {
	v := tensor.New([]int{2, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Transpose(v, 0, 1)

	fmt.Println(w.Shape)
	fmt.Println(w.At(0, 0), w.At(0, 1), w.At(0, 2))
	fmt.Println(w.At(1, 0), w.At(1, 1), w.At(1, 2))

	// Output:
	// [2 3]
	// 1 2 3
	// 4 5 6
}

func ExampleTranspose_axes10() {
	v := tensor.New([]int{2, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Transpose(v, 1, 0)

	fmt.Println(w.Shape)
	fmt.Println(w.At(0, 0), w.At(0, 1))
	fmt.Println(w.At(1, 0), w.At(1, 1))
	fmt.Println(w.At(2, 0), w.At(2, 1))

	// Output:
	// [3 2]
	// 1 4
	// 2 5
	// 3 6
}

func ExampleTranspose_plus() {
	v := tensor.New([]int{2, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})

	// [1, 4]
	// [2, 5]
	// [3, 6]
	x0 := tensor.Transpose(v, 1, 0)
	x1 := tensor.New([]int{3, 2}, []int{
		1, 2,
		3, 4,
		5, 6,
	})

	z := tensor.Add(x0, x1)

	fmt.Println(z.Shape)
	fmt.Println(z.At(0, 0), z.At(0, 1))
	fmt.Println(z.At(1, 0), z.At(1, 1))
	fmt.Println(z.At(2, 0), z.At(2, 1))

	// Output:
	// [3 2]
	// 2 6
	// 5 9
	// 8 12
}

func ExampleTranspose_minus12() {
	v := tensor.New([]int{2, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Transpose(v, -1, -2)

	fmt.Println(w.Shape)
	fmt.Println(w.At(0, 0), w.At(0, 1))
	fmt.Println(w.At(1, 0), w.At(1, 1))
	fmt.Println(w.At(2, 0), w.At(2, 1))

	// Output:
	// [3 2]
	// 1 4
	// 2 5
	// 3 6
}

func ExampleTranspose_ndim0() {
	v := tensor.New([]int{}, []int{1})
	w := tensor.Transpose(v)

	fmt.Println(w.Shape)
	fmt.Println(w.At())

	// Output:
	// []
	// 1
}

func ExampleSqueeze() {
	v := tensor.New([]int{1, 2, 1, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Squeeze(v)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 3]
	// [1 2 3 4 5 6]
}

func ExampleSqueeze_axis0() {
	v := tensor.New([]int{1, 2, 1, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Squeeze(v, 0)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 1 3]
	// [1 2 3 4 5 6]
}

func ExampleSqueeze_axis1() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println(r)
			return
		}

		panic("unexpected panic for index")
	}()

	v := tensor.New([]int{1, 2, 1, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	_ = tensor.Squeeze(v, 1)

	panic("unreachable")

	// Output:
	// axis=1 is not 1 (shape [1 2 1 3])
}

func ExampleSqueeze_axis2() {
	v := tensor.New([]int{1, 2, 1, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Squeeze(v, 2)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [1 2 3]
	// [1 2 3 4 5 6]
}

func ExampleSqueeze_minus2() {
	v := tensor.New([]int{1, 2, 1, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Squeeze(v, -2)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [1 2 3]
	// [1 2 3 4 5 6]
}

func ExampleSqueeze_minus4() {
	v := tensor.New([]int{1, 2, 1, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Squeeze(v, -4)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 1 3]
	// [1 2 3 4 5 6]
}

func ExampleExpand() {
	v := tensor.New([]int{2, 1, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Expand(v, 0)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [1 2 1 3]
	// [1 2 3 4 5 6]
}

func ExampleExpand_axis1() {
	v := tensor.New([]int{2, 1, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Expand(v, 1)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 1 1 3]
	// [1 2 3 4 5 6]
}

func ExampleExpand_axis2() {
	v := tensor.New([]int{2, 1, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Expand(v, 3)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 1 3 1]
	// [1 2 3 4 5 6]
}
func ExampleExpand_minus1() {
	v := tensor.New([]int{2, 1, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Expand(v, -1)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 1 3 1]
	// [1 2 3 4 5 6]
}

func ExampleExpand_minus2() {
	v := tensor.New([]int{2, 1, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	w := tensor.Expand(v, -2)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 1 1 3]
	// [1 2 3 4 5 6]
}

func ExampleBroadcastTo_axis0() {
	v := tensor.New([]int{1, 4}, []int{1, 2, 3, 4})
	w := tensor.BroadcastTo(v, 2, 4)

	fmt.Println(w.Shape)
	fmt.Println(w.At(0, 0), w.At(0, 1), w.At(0, 2), w.At(0, 3))
	fmt.Println(w.At(1, 0), w.At(1, 1), w.At(1, 2), w.At(1, 3))

	// Output:
	// [2 4]
	// 1 2 3 4
	// 1 2 3 4
}

func ExampleBroadcastTo_axis1() {
	v := tensor.New([]int{4, 1}, []int{
		1,
		2,
		3,
		4,
	})
	w := tensor.BroadcastTo(v, 4, 2)

	fmt.Println(w.Shape)
	fmt.Println(w.At(0, 0), w.At(0, 1))
	fmt.Println(w.At(1, 0), w.At(1, 1))
	fmt.Println(w.At(2, 0), w.At(2, 1))
	fmt.Println(w.At(3, 0), w.At(3, 1))

	// Output:
	// [4 2]
	// 1 1
	// 2 2
	// 3 3
	// 4 4
}

func ExampleBroadcastTo_same() {
	v := tensor.New([]int{1, 4}, []int{1, 2, 3, 4})
	w := tensor.BroadcastTo(v, 1, 4)

	fmt.Println(w.Shape)
	fmt.Println(w.At(0, 0), w.At(0, 1), w.At(0, 2), w.At(0, 3))

	// Output:
	// [1 4]
	// 1 2 3 4
}

func ExampleBroadcastTo_add() {
	v := tensor.New([]int{1, 4}, []int{1, 2, 3, 4})
	w := tensor.BroadcastTo(v, 2, 2, 4)

	fmt.Println(w.Shape)
	fmt.Println(w.At(0, 0, 0), w.At(0, 0, 1), w.At(0, 0, 2), w.At(0, 0, 3))
	fmt.Println(w.At(0, 1, 0), w.At(0, 1, 1), w.At(0, 1, 2), w.At(0, 1, 3))
	fmt.Println(w.At(1, 0, 0), w.At(1, 0, 1), w.At(1, 0, 2), w.At(1, 0, 3))
	fmt.Println(w.At(1, 1, 0), w.At(1, 1, 1), w.At(1, 1, 2), w.At(1, 1, 3))

	// Output:
	// [2 2 4]
	// 1 2 3 4
	// 1 2 3 4
	// 1 2 3 4
	// 1 2 3 4
}

func ExampleBroadcastTo_multi() {
	v := tensor.New([]int{1, 2, 1}, []int{1, 2})
	w := tensor.BroadcastTo(v, 3, 2, 4)

	fmt.Println(w.Shape)
	fmt.Println(w.At(0, 0, 0), w.At(0, 0, 1), w.At(0, 0, 2), w.At(0, 0, 3))
	fmt.Println(w.At(0, 1, 0), w.At(0, 1, 1), w.At(0, 1, 2), w.At(0, 1, 3))
	fmt.Println(w.At(1, 0, 0), w.At(1, 0, 1), w.At(1, 0, 2), w.At(1, 0, 3))
	fmt.Println(w.At(1, 1, 0), w.At(1, 1, 1), w.At(1, 1, 2), w.At(1, 1, 3))
	fmt.Println(w.At(2, 0, 0), w.At(2, 0, 1), w.At(2, 0, 2), w.At(2, 0, 3))
	fmt.Println(w.At(2, 1, 0), w.At(2, 1, 1), w.At(2, 1, 2), w.At(2, 1, 3))

	// Output:
	// [3 2 4]
	// 1 1 1 1
	// 2 2 2 2
	// 1 1 1 1
	// 2 2 2 2
	// 1 1 1 1
	// 2 2 2 2
}

func ExampleBroadcastTo_scalar() {
	v := tensor.New(nil, []float64{1.5})
	w := tensor.BroadcastTo(v, 2, 2)

	fmt.Println(w.Shape)
	fmt.Println(w.At(0, 0), w.At(0, 1))
	fmt.Println(w.At(1, 0), w.At(1, 1))

	// Output:
	// [2 2]
	// 1.5 1.5
	// 1.5 1.5
}

func ExampleMatMul() {
	a := tensor.New([]int{2, 3}, []int{
		1, 2, 3,
		4, 5, 6,
	})
	b := tensor.New([]int{3, 2}, []int{
		7, 8,
		9, 10,
		11, 12,
	})
	c := tensor.MatMul(a, b)

	fmt.Println(c.Shape)
	fmt.Println(c.Data)

	// Output:
	// [2 2]
	// [58 64 139 154]
}

func ExampleMatMul_batch() {
	a := tensor.New([]int{2, 2, 3}, []int{
		1, 2, 3,
		4, 5, 6,

		7, 8, 9,
		10, 11, 12,
	})
	b := tensor.New([]int{2, 3, 2}, []int{
		7, 8,
		9, 10,
		11, 12,

		13, 14,
		15, 16,
		17, 18,
	})
	c := tensor.MatMul(a, b)

	fmt.Println(c.Shape)
	fmt.Println(c.Data)

	// Output:
	// [2 2 2]
	// [58 64 139 154 364 388 499 532]
}

func ExampleMatMul_invalid() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println(r)
			return
		}

		panic("unexpected panic for index")
	}()

	a := tensor.New([]int{2, 2, 2}, []int{
		1, 2,
		4, 5,

		6, 7,
		8, 9,
	})
	b := tensor.New([]int{2, 3, 2}, []int{
		7, 8,
		9, 10,
		11, 12,

		13, 14,
		15, 16,
		17, 18,
	})

	_ = tensor.MatMul(a, b)
	panic("unreachable")

	// Output:
	// shapes [2 2 2] and [2 3 2] are not aligned for matmul
}

func ExampleBroadcastTo_backward() {
	x := tensor.New([]int{1, 2, 2}, []float64{
		1, 2,
		3, 4,
	})
	y := tensor.BroadcastTo(x, 2, 2, 2)
	z := tensor.Sum(y, 0)

	fmt.Println(z.Shape)
	fmt.Println(z.Data)

	// Output:
	// [2 2]
	// [2 4 6 8]
}

func ExampleSum_backward() {
	x := tensor.New([]int{2, 2, 2}, []float64{
		1, 2,
		3, 4,

		1, 2,
		3, 4,
	})
	y := tensor.Sum(x, 0)
	z := tensor.BroadcastTo(y, 2, 2, 2)

	fmt.Println(z.Shape)
	fmt.Println(z.Data)

	// Output:
	// [2 2 2]
	// [2 4 6 8 2 4 6 8]
}

func TestTake(t *testing.T) {
	cases := []struct {
		v       *tensor.Tensor[int]
		indices []int
		axis    int
		want    *tensor.Tensor[int]
	}{
		{
			v: tensor.New([]int{3, 2}, []int{
				10, 11,
				20, 21,
				30, 31,
			}),
			axis:    0,
			indices: []int{0, 2},
			want: tensor.New([]int{2, 2}, []int{
				10, 11,
				30, 31,
			}),
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			axis:    1,
			indices: []int{2, 0},
			want: tensor.New([]int{2, 2}, []int{
				3, 1,
				6, 4,
			}),
		},
		{
			v: tensor.New([]int{2, 2, 3},
				[]int{
					1, 2, 3,
					4, 5, 6,

					10, 11, 12,
					13, 14, 15,
				}),
			axis:    2,
			indices: []int{0, 2},
			want: tensor.New([]int{2, 2, 2}, []int{
				1, 3,
				4, 6,

				10, 12,
				13, 15,
			}),
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			axis:    1,
			indices: []int{1, 1, 2},
			want: tensor.New([]int{2, 3}, []int{
				2, 2, 3,
				5, 5, 6,
			}),
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			axis:    -1,
			indices: []int{1},
			want:    tensor.New([]int{2, 1}, []int{2, 5}),
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			axis:    1,
			indices: []int{-1, 0},
			want: tensor.New([]int{2, 2}, []int{
				3, 1,
				6, 4,
			}),
		},
	}

	for _, c := range cases {
		got := tensor.Take(c.v, c.indices, c.axis)
		if !tensor.Equal(got, c.want) {
			t.Errorf("got=%v, want=%v", got.Data, c.want.Data)
		}
	}
}

func TestScatterAdd(t *testing.T) {
	cases := []struct {
		v       *tensor.Tensor[int]
		w       *tensor.Tensor[int]
		indices []int
		axis    int
		want    *tensor.Tensor[int]
	}{
		{
			v: tensor.New([]int{3, 2}, []int{
				10, 11,
				20, 21,
				30, 31,
			}),
			w: tensor.New([]int{2, 2}, []int{
				1, 1,
				2, 2,
			}),
			axis:    0,
			indices: []int{0, 2},
			want: tensor.New([]int{3, 2}, []int{
				11, 12,
				20, 21,
				32, 33,
			}),
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			w: tensor.New([]int{2, 2}, []int{
				10, 20,
				30, 40,
			}),
			axis:    1,
			indices: []int{2, 0},
			want: tensor.New([]int{2, 3}, []int{
				21, 2, 13,
				44, 5, 36,
			}),
		},
		{
			v: tensor.New([]int{2, 2, 3},
				[]int{
					1, 2, 3,
					4, 5, 6,

					10, 11, 12,
					13, 14, 15,
				}),
			w: tensor.New([]int{2, 2, 2}, []int{
				100, 200,
				300, 400,

				1000, 1100,
				1200, 1300,
			}),
			axis:    2,
			indices: []int{0, 2},
			want: tensor.New([]int{2, 2, 3}, []int{
				101, 2, 203,
				304, 5, 406,

				1010, 11, 1112,
				1213, 14, 1315,
			}),
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			w: tensor.New([]int{2, 3}, []int{
				10, 20, 30,
				40, 50, 60,
			}),
			axis:    1,
			indices: []int{1, 1, 2},
			want: tensor.New([]int{2, 3}, []int{
				1, 32, 33,
				4, 95, 66,
			}),
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			w: tensor.New([]int{2, 1}, []int{
				10, 20,
			}),
			axis:    -1,
			indices: []int{1},
			want: tensor.New([]int{2, 3}, []int{
				1, 12, 3,
				4, 25, 6,
			}),
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			w: tensor.New([]int{2, 2}, []int{
				10, 20,
				30, 40,
			}),
			axis:    1,
			indices: []int{-1, 0},
			want: tensor.New([]int{2, 3}, []int{
				21, 2, 13,
				44, 5, 36,
			}),
		},
	}

	for _, c := range cases {
		c.v.ScatterAdd(c.w, c.indices, c.axis)
		if !tensor.Equal(c.v, c.want) {
			t.Errorf("got=%v, want=%v", c.v.Data, c.want.Data)
		}
	}
}

func TestEqual(t *testing.T) {
	cases := []struct {
		v, w *tensor.Tensor[int]
		want bool
	}{
		{
			v:    tensor.New([]int{2, 3}, []int{1, 2, 3, 4, 5, 6}),
			w:    tensor.New([]int{2, 3}, []int{1, 2, 3, 4, 5, 6}),
			want: true,
		},
		{
			v:    tensor.New([]int{2, 3}, []int{1, 2, 3, 4, 5, 6}),
			w:    tensor.New([]int{2, 3}, []int{1, 2, 3, 4, 5, 0}),
			want: false,
		},
		{
			v:    tensor.New([]int{2, 3}, []int{1, 2, 3, 4, 5, 6}),
			w:    tensor.New([]int{3, 2}, []int{1, 2, 3, 4, 5, 6}),
			want: false,
		},
	}

	for _, c := range cases {
		got := tensor.Equal(c.v, c.w)
		if got != c.want {
			t.Errorf("got=%v, want=%v", got, c.want)
		}
	}
}

func TestIsClose(t *testing.T) {
	cases := []struct {
		v, w *tensor.Tensor[float64]
		want bool
	}{
		{
			v:    tensor.New([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6}),
			w:    tensor.New([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6}),
			want: true,
		},
		{
			v:    tensor.New([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6}),
			w:    tensor.New([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6.0000001}),
			want: true,
		},
		{
			v:    tensor.New([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6}),
			w:    tensor.New([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6.1}),
			want: false,
		},
		{
			v:    tensor.New([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6}),
			w:    tensor.New([]int{3, 2}, []float64{1, 2, 3, 4, 5, 6}),
			want: false,
		},
	}

	for _, c := range cases {
		got := tensor.IsClose(c.v, c.w, 1e-8, 1e-5)
		if got != c.want {
			t.Errorf("got=%v, want=%v", got, c.want)
		}
	}
}

func TestEqualShape(t *testing.T) {
	cases := []struct {
		a, b []int
		want bool
	}{
		{
			a:    []int{2, 3},
			b:    []int{2, 3},
			want: true,
		},
		{
			a:    []int{2, 3},
			b:    []int{3, 2},
			want: false,
		},
		{
			a:    []int{2, 3},
			b:    []int{2, 3, 1},
			want: false,
		},
	}

	for _, c := range cases {
		got := tensor.EqualShape(c.a, c.b)
		if got != c.want {
			t.Errorf("got=%v, want=%v", got, c.want)
		}
	}
}

func TestArgmax(t *testing.T) {
	cases := []struct {
		in   *tensor.Tensor[int]
		axis int
		out  *tensor.Tensor[int]
	}{
		{
			in: tensor.New([]int{5}, []int{
				1, 7, 3, 7, 2,
			}),
			axis: 0,
			out:  tensor.New([]int{}, []int{1}),
		},
		{
			in: tensor.New([]int{2, 4}, []int{
				1, 2, 3, 4,
				4, 3, 2, 1,
			}),
			axis: 0,
			out:  tensor.New([]int{4}, []int{1, 1, 0, 0}),
		},
		{
			in: tensor.New([]int{2, 4}, []int{
				1, 2, 3, 4,
				4, 3, 2, 1,
			}),
			axis: 1,
			out:  tensor.New([]int{2}, []int{3, 0}),
		},
		{
			in: tensor.New([]int{2, 2, 3}, []int{
				1, 5, 2,
				9, 3, 0,
				//
				4, 6, 7,
				1, 2, 8,
			}),
			axis: 2,
			out:  tensor.New([]int{2, 2}, []int{1, 0, 2, 2}),
		},
		{
			in: tensor.New([]int{3}, []int{
				5, 5, 5,
			}),
			axis: 0,
			out:  tensor.New([]int{}, []int{0}),
		},
		{
			in: tensor.New([]int{4}, []int{
				-10, -5, -20, -5,
			}),
			axis: 0,
			out:  tensor.New([]int{}, []int{1}),
		},
		{
			in: tensor.New([]int{2, 4}, []int{
				1, 2, 3, 4,
				4, 3, 2, 1,
			}),
			axis: -1,
			out:  tensor.New([]int{2}, []int{3, 0}),
		},
	}

	for _, c := range cases {
		got := tensor.Argmax(c.in, c.axis)
		if tensor.Equal(got, c.out) {
			continue
		}

		t.Errorf("axis=%d, got=%v(%v), want=%v(%v)", c.axis, got.Data, got.Shape, c.out.Data, c.out.Shape)
	}
}

func TestMatMul(t *testing.T) {
	cases := []struct {
		a, b *tensor.Tensor[int]
		out  *tensor.Tensor[int]
	}{
		{
			a: tensor.New([]int{2, 2, 2, 3}, []int{
				1, 2, 3,
				4, 5, 6,

				7, 8, 9,
				10, 11, 12,

				2, 1, 0,
				0, 1, 2,

				3, 3, 3,
				1, 2, 3,
			}),
			b: tensor.New([]int{2, 2, 3, 2}, []int{
				1, 0,
				0, 1,
				1, 1,

				2, 2,
				3, 3,
				4, 4,

				1, 2,
				3, 4,
				5, 6,

				0, 1,
				1, 0,
				1, 1,
			}),
			out: tensor.New([]int{2, 2, 2, 2}, []int{
				4, 5,
				10, 11,

				74, 74,
				101, 101,

				5, 8,
				13, 16,

				6, 6,
				5, 4,
			}),
		},
		{
			a: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			b: tensor.New([]int{3, 2}, []int{
				0, 0,
				0, 0,
				0, 0,
			}),
			out: tensor.New([]int{2, 2}, []int{
				0, 0,
				0, 0,
			}),
		},
		{
			a: tensor.New([]int{1, 4}, []int{
				1, 2, 3, 4,
			}),
			b: tensor.New([]int{4, 1}, []int{
				1,
				2,
				3,
				4,
			}),
			out: tensor.New([]int{1, 1}, []int{
				30,
			}),
		},
		{
			// broadcast
			a: tensor.New([]int{1, 2, 2}, []int{
				1, 2,
				3, 4,
			}),
			b: tensor.New([]int{2, 2, 2}, []int{
				1, 2,
				3, 4,

				1, 2,
				3, 4,
			}),
			out: tensor.New([]int{2, 2, 2}, []int{
				7, 10,
				15, 22,

				7, 10,
				15, 22,
			}),
		},
	}

	for _, c := range cases {
		got := tensor.MatMul(c.a, c.b)
		if tensor.Equal(got, c.out) {
			continue
		}

		t.Errorf("got=%v(%v), want=%v(%v)", got.Data, got.Shape, c.out.Data, c.out.Shape)
	}
}

func TestRavel(t *testing.T) {
	cases := []struct {
		v     *tensor.Tensor[int]
		coord []int
		want  int
	}{
		{v: tensor.Zero[int](2, 3), coord: []int{0, 0}, want: 0},
		{v: tensor.Zero[int](2, 3), coord: []int{0, 1}, want: 1},
		{v: tensor.Zero[int](2, 3), coord: []int{0, 2}, want: 2},
		{v: tensor.Zero[int](2, 3), coord: []int{1, 0}, want: 3},
		{v: tensor.Zero[int](2, 3), coord: []int{1, 1}, want: 4},
		{v: tensor.Zero[int](2, 3), coord: []int{1, 2}, want: 5},
	}

	for _, c := range cases {
		got := tensor.Ravel(c.v, c.coord...)
		if got == c.want {
			continue
		}

		t.Errorf("coord=%v, got=%v, want=%v", c.coord, got, c.want)
	}
}

func TestUnravel(t *testing.T) {
	cases := []struct {
		v     *tensor.Tensor[int]
		index int
		want  []int
	}{
		{v: tensor.Zero[int](), index: 0, want: []int{}},
		{v: tensor.Zero[int](5), index: 0, want: []int{0}},
		{v: tensor.Zero[int](5), index: 4, want: []int{4}},
		{v: tensor.Zero[int](2, 3), index: 0, want: []int{0, 0}},
		{v: tensor.Zero[int](2, 3), index: 1, want: []int{0, 1}},
		{v: tensor.Zero[int](2, 3), index: 2, want: []int{0, 2}},
		{v: tensor.Zero[int](2, 3), index: 3, want: []int{1, 0}},
		{v: tensor.Zero[int](2, 3), index: 4, want: []int{1, 1}},
		{v: tensor.Zero[int](2, 3), index: 5, want: []int{1, 2}},
	}

	for _, c := range cases {
		got := tensor.Unravel(c.v, c.index)
		if reflect.DeepEqual(got, c.want) {
			continue
		}

		t.Errorf("index=%v, got=%v, want=%v", c.index, got, c.want)
	}
}

func TestStride(t *testing.T) {
	cases := []struct {
		shape []int
		want  []int
	}{
		{shape: []int{}, want: nil},
		{shape: []int{5}, want: []int{1}},
		{shape: []int{2, 3}, want: []int{3, 1}},
		{shape: []int{2, 3, 4}, want: []int{12, 4, 1}},
		{shape: []int{4, 1, 2}, want: []int{2, 2, 1}},
	}

	for _, c := range cases {
		got := tensor.Stride(c.shape...)
		if reflect.DeepEqual(got, c.want) {
			continue
		}

		t.Errorf("shape=%v, got=%v, want=%v", c.shape, got, c.want)
	}
}

func TestValidate(t *testing.T) {
	cases := []struct {
		v      *tensor.Tensor[int]
		axes   []int
		hasErr bool
	}{
		{v: tensor.Zero[int](), axes: []int{0, 1}, hasErr: true},
		{v: tensor.Zero[int](2, 3), axes: []int{0, 0}, hasErr: true},
	}

	for _, c := range cases {
		if _, _, err := tensor.Validate(c.v, c.axes...); err != nil {
			if c.hasErr {
				continue
			}

			t.Errorf("unexpected error for axes %v: %v", c.axes, err)
		}
	}
}

func TestBroadcastShape(t *testing.T) {
	cases := []struct {
		s0       []int
		s1       []int
		keepLast int
		want0    []int
		want1    []int
		hasErr   bool
	}{
		{s0: []int{2, 3}, s1: []int{2, 3}, want0: []int{2, 3}, want1: []int{2, 3}},
		{s0: []int{2, 3}, s1: []int{1, 3}, want0: []int{2, 3}, want1: []int{2, 3}},
		{s0: []int{2, 3}, s1: []int{2, 1}, want0: []int{2, 3}, want1: []int{2, 3}},
		{s0: []int{2, 3}, s1: []int{1, 1}, want0: []int{2, 3}, want1: []int{2, 3}},
		{s0: []int{1, 3}, s1: []int{2, 3}, want0: []int{2, 3}, want1: []int{2, 3}},
		{s0: []int{2, 1}, s1: []int{2, 3}, want0: []int{2, 3}, want1: []int{2, 3}},
		{s0: []int{1, 1}, s1: []int{2, 3}, want0: []int{2, 3}, want1: []int{2, 3}},
		{s0: []int{1, 3, 5}, s1: []int{1, 3, 1}, want0: []int{1, 3, 5}, want1: []int{1, 3, 5}},
		{s0: []int{2, 1, 3}, s1: []int{1, 4, 1}, want0: []int{2, 4, 3}, want1: []int{2, 4, 3}},
		{s0: []int{8, 1, 6, 1}, s1: []int{7, 1, 5}, want0: []int{8, 7, 6, 5}, want1: []int{8, 7, 6, 5}},
		{s0: nil, s1: []int{2, 3}, want0: []int{2, 3}, want1: []int{2, 3}},
		{s0: []int{2, 3}, s1: nil, want0: []int{2, 3}, want1: []int{2, 3}},
		{s0: []int{}, s1: []int{2, 3}, want0: []int{2, 3}, want1: []int{2, 3}},
		{s0: []int{2, 3}, s1: []int{}, want0: []int{2, 3}, want1: []int{2, 3}},
		{s0: nil, s1: nil, want0: []int{}, want1: []int{}},
		{s0: []int{}, s1: []int{}, want0: []int{}, want1: []int{}},
		{s0: []int{3}, s1: []int{2, 3}, want0: []int{2, 3}, want1: []int{2, 3}},
		// keepLast=2
		{s0: []int{1, 1, 3}, s1: []int{2, 3, 1}, keepLast: 2, want0: []int{2, 1, 3}, want1: []int{2, 3, 1}},
		{s0: []int{4, 1, 3}, s1: []int{1, 3, 1}, keepLast: 2, want0: []int{4, 1, 3}, want1: []int{4, 3, 1}},
		{s0: []int{2, 3}, s1: []int{2, 3}, keepLast: 2, want0: []int{2, 3}, want1: []int{2, 3}},
		{s0: []int{1, 2, 1, 3}, s1: []int{2, 1, 3, 1}, keepLast: 2, want0: []int{2, 2, 1, 3}, want1: []int{2, 2, 3, 1}},
		// error
		{s0: []int{4, 5}, s1: []int{2, 3}, hasErr: true},
		{s0: []int{3}, s1: []int{2, 3, 4}, hasErr: true},
		{s0: []int{2, 3, 4}, s1: []int{2, 4, 4}, hasErr: true},
		{s0: []int{3, 4}, s1: []int{4, 3}, hasErr: true},
		{s0: []int{2, 3, 4}, s1: []int{3, 2, 4}, keepLast: 2, hasErr: true},
		{s0: []int{4, 1, 3, 4}, s1: []int{2, 3, 2, 4}, keepLast: 2, hasErr: true},
		{s0: []int{3, 3}, s1: []int{1, 1, 4, 1}, keepLast: 3, hasErr: true},
		{s0: []int{1, 1, 4, 1}, s1: []int{3, 3}, keepLast: 3, hasErr: true},
	}

	for _, c := range cases {
		got0, got1, err := tensor.BroadcastShape(c.s0, c.s1, c.keepLast)
		if err != nil {
			if c.hasErr {
				continue
			}

			t.Errorf("unexpected error for shapes %v and %v: %v", c.s0, c.s1, err)
			continue
		}

		if !reflect.DeepEqual(got0, c.want0) {
			t.Errorf("s0=%v, got0=%v, want0=%v", c.s0, got0, c.want0)
		}

		if !reflect.DeepEqual(got1, c.want1) {
			t.Errorf("s1=%v, got1=%v, want1=%v", c.s1, got1, c.want1)
		}
	}
}

func TestMean_invalid(t *testing.T) {
	cases := []struct {
		v    *tensor.Tensor[float64]
		axes []int
	}{
		{v: tensor.Zero[float64](1, 4), axes: []int{10}},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					return
				}

				t.Errorf("unexpected panic for coord %v", c.axes)
			}()

			_ = tensor.Mean(c.v, c.axes...)
			t.Fail()
		}()
	}
}

func TestTake_invalid(t *testing.T) {
	cases := []struct {
		v       *tensor.Tensor[int]
		indices []int
		axis    int
	}{
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			axis:    2,
			indices: []int{0},
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			axis:    -3,
			indices: []int{0},
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			axis:    1,
			indices: []int{3},
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			axis:    1,
			indices: []int{-4},
		},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					return
				}

				t.Errorf("unexpected panic for axis %d and indices %v", c.axis, c.indices)
			}()

			_ = tensor.Take(c.v, c.indices, c.axis)
			t.Fail()
		}()
	}
}

func TestScatterAdd_invalid(t *testing.T) {
	cases := []struct {
		v, w    *tensor.Tensor[int]
		indices []int
		axis    int
	}{
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			w: tensor.New([]int{2, 2}, []int{
				10, 20,
				30, 40,
			}),
			axis:    2,
			indices: []int{0, 1},
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			w: tensor.New([]int{2, 2}, []int{
				10, 20,
				30, 40,
			}),
			axis:    -3,
			indices: []int{0, 1},
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			axis:    2,
			indices: []int{},
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			w: tensor.New([]int{2, 2}, []int{
				10, 20,
				30, 40,
			}),
			axis:    1,
			indices: []int{3, 0},
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			w: tensor.New([]int{2, 2}, []int{
				10, 20,
				30, 40,
			}),
			axis:    1,
			indices: []int{-4, 0},
		},
		{
			v: tensor.New([]int{2, 3}, []int{
				1, 2, 3,
				4, 5, 6,
			}),
			w: tensor.New([]int{2, 3}, []int{
				10, 20, 30,
				40, 50, 60,
			}),
			axis:    1,
			indices: []int{1, 1, 3},
		},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					return
				}

				t.Errorf("unexpected panic for axis %d and indices %v", c.axis, c.indices)
			}()

			c.v.ScatterAdd(c.w, c.indices, c.axis)
			t.Fail()
		}()
	}
}

func TestArgmax_invalid(t *testing.T) {
	cases := []struct {
		in   *tensor.Tensor[int]
		axis int
	}{
		{in: tensor.New([]int{2, 3}, []int{
			1, 2, 3,
			4, 5, 6,
		}), axis: 2},
		{in: tensor.New([]int{2, 3}, []int{
			1, 2, 3,
			4, 5, 6,
		}), axis: -3},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					return
				}

				t.Errorf("unexpected panic for axis %d", c.axis)
			}()

			_ = tensor.Argmax(c.in, c.axis)
			t.Fail()
		}()
	}
}

func TestTranspose_invalid(t *testing.T) {
	cases := []struct {
		v    *tensor.Tensor[int]
		axes []int
	}{
		{v: tensor.Zero[int](2, 3), axes: []int{0, 0}},
		{v: tensor.Zero[int](2, 3), axes: []int{0, -3}},
		{v: tensor.Zero[int](2, 3), axes: []int{0, 3}},
		{v: tensor.Zero[int](2, 3), axes: []int{0, 1, 2}},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					return
				}

				t.Errorf("unexpected panic for axes %v", c.axes)
			}()

			_ = tensor.Transpose(c.v, c.axes...)
			t.Fail()
		}()
	}
}

func TestSqueeze_invalid(t *testing.T) {
	cases := []struct {
		v    *tensor.Tensor[int]
		axes []int
	}{
		{v: tensor.Zero[int](1, 3), axes: []int{0, -10}},
		{v: tensor.Zero[int](1, 3), axes: []int{0, 10}},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					return
				}

				t.Errorf("unexpected panic for axes %v", c.axes)
			}()

			_ = tensor.Squeeze(c.v, c.axes...)
			t.Fail()
		}()
	}
}

func TestBroadcastTo_invalid(t *testing.T) {
	cases := []struct {
		v     *tensor.Tensor[int]
		shape []int
	}{
		{v: tensor.Zero[int](2, 3), shape: []int{2}},
		{v: tensor.Zero[int](2, 3), shape: []int{3, 3}},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					return
				}

				t.Errorf("unexpected panic for shape %v", c.shape)
			}()

			_ = tensor.BroadcastTo(c.v, c.shape...)
			t.Fail()
		}()
	}
}

func TestExpand_invalid(t *testing.T) {
	cases := []struct {
		v    *tensor.Tensor[int]
		axis int
	}{
		{v: tensor.Zero[int](2, 3), axis: -4},
		{v: tensor.Zero[int](2, 3), axis: 3},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					return
				}

				t.Errorf("unexpected panic for axis %v", c.axis)
			}()

			_ = tensor.Expand(c.v, c.axis)
			t.Fail()
		}()
	}
}

func TestBroadcast_invalid(t *testing.T) {
	cases := []struct {
		v1, v2 *tensor.Tensor[int]
	}{
		{v1: tensor.Zero[int](2, 3), v2: tensor.Zero[int](3, 2)},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					return
				}

				t.Errorf("unexpected panic for shapes %v and %v", c.v1.Shape, c.v2.Shape)
			}()

			_, _ = tensor.Broadcast(c.v1, c.v2)
			t.Fail()
		}()
	}
}

func TestRavel_invalid(t *testing.T) {
	cases := []struct {
		v     *tensor.Tensor[int]
		coord []int
	}{
		{v: tensor.Zero[int](2, 3), coord: []int{1}},
		{v: tensor.Zero[int](2, 3), coord: []int{2, 3, 4}},
		{v: tensor.Zero[int](2, 3), coord: []int{-1, 0}},
		{v: tensor.Zero[int](2, 3), coord: []int{2, 0}},
		{v: tensor.Zero[int](2, 3), coord: []int{0, 3}},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					return
				}

				t.Errorf("unexpected panic for coord %v", c.coord)
			}()

			_ = tensor.Ravel(c.v, c.coord...)
			t.Fail()
		}()
	}
}

func TestReduce_invalid(t *testing.T) {
	cases := []struct {
		v     *tensor.Tensor[int]
		coord []int
	}{
		{v: tensor.Zero[int](1, 4), coord: []int{10}},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					return
				}

				t.Errorf("unexpected panic for coord %v", c.coord)
			}()

			_ = tensor.Reduce(c.v, 0, func(a, b int) int { return a + b }, c.coord...)
			t.Fail()
		}()
	}
}

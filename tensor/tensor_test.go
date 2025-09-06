package tensor_test

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/itsubaki/autograd/tensor"
)

func Example() {
	v := tensor.Zero(2, 3)

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
	v := tensor.Zero(2, 3)
	w := tensor.ZeroLike(v)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 3]
	// [0 0 0 0 0 0]
}

func ExampleOneLike() {
	v := tensor.Zero(2, 3, 4)
	w := tensor.OneLike(v)
	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2 3 4]
	// [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
}

func ExampleAddC() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.AddC(10, v)

	fmt.Println(w.Data)

	// Output:
	// [11 12 13 14]
}

func ExampleSubC() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.SubC(10, v)

	fmt.Println(w.Data)

	// Output:
	// [9 8 7 6]
}

func ExampleMulC() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.MulC(10, v)

	fmt.Println(w.Data)

	// Output:
	// [10 20 30 40]
}

func ExamplePow() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Pow(v, 3)

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

func ExampleSum() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Sum(v)

	fmt.Println(w.At())

	// Output:
	// 10
}

func ExampleSum_axis0() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Sum(v, 0)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [4 6]
}

func ExampleSum_axis1() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Sum(v, 1)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [3 7]
}

func ExampleSum_axisAll() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Sum(v, 0, 1)

	fmt.Println(w.Data)

	// Output:
	// [10]
}

func ExampleMax() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Max(v)

	fmt.Println(w.At())

	// Output:
	// 4
}

func ExampleMax_axis0() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Max(v, 0)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [3 4]
}

func ExampleMax_axis1() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Max(v, 1)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [2 4]
}

func ExampleMax_axisAll() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Max(v, 0, 1)

	fmt.Println(w.At())

	// Output:
	// 4
}

func ExampleMin() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Min(v)

	fmt.Println(w.At())

	// Output:
	// 1
}

func ExampleMin_axis0() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Min(v, 0)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [1 2]
}

func ExampleMin_axis1() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Min(v, 1)

	fmt.Println(w.Shape)
	fmt.Println(w.Data)

	// Output:
	// [2]
	// [1 3]
}

func ExampleMin_axisAll() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.Min(v, 0, 1)

	fmt.Println(w.At())

	// Output:
	// 1
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

func ExampleTensor_Clone() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := v.Clone()
	w.Set([]int{0, 0}, 10)

	fmt.Println(v.Data)
	fmt.Println(w.Data)

	// Output:
	// [1 2 3 4]
	// [10 2 3 4]
}

func ExampleTensor_Reshape() {
	v := tensor.New([]int{2, 2}, []float64{1, 2, 3, 4})
	w := v.Reshape(1, 4)

	fmt.Println(w.Shape)
	fmt.Println(w.At(0, 0), w.At(0, 1), w.At(0, 2), w.At(0, 3))

	// Output:
	// [1 4]
	// 1 2 3 4
}

func TestIndex(t *testing.T) {
	cases := []struct {
		v     *tensor.Tensor
		index []int
		want  int
	}{
		{v: tensor.Zero(2, 3), index: []int{0, 0}, want: 0},
		{v: tensor.Zero(2, 3), index: []int{0, 1}, want: 1},
		{v: tensor.Zero(2, 3), index: []int{0, 2}, want: 2},
		{v: tensor.Zero(2, 3), index: []int{1, 0}, want: 3},
		{v: tensor.Zero(2, 3), index: []int{1, 1}, want: 4},
		{v: tensor.Zero(2, 3), index: []int{1, 2}, want: 5},
	}

	for _, c := range cases {
		got := tensor.Index(c.v, c.index...)
		if got == c.want {
			continue
		}

		t.Errorf("Index(%v) = %d, want %d", c.index, got, c.want)
	}
}

func TestIndex_outOfBounds(t *testing.T) {
	cases := []struct {
		v     *tensor.Tensor
		index []int
	}{
		{v: tensor.Zero(2, 3), index: []int{-1, 0}},
		{v: tensor.Zero(2, 3), index: []int{2, 0}},
		{v: tensor.Zero(2, 3), index: []int{0, 3}},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					return
				}

				t.Errorf("unexpected panic for index %v", c.index)
			}()

			_ = tensor.Index(c.v, c.index...)
			t.Fail()
		}()
	}
}

func TestStride(t *testing.T) {
	cases := []struct {
		shape []int
		want  []int
	}{
		{shape: []int{}, want: nil},                    // scalar
		{shape: []int{5}, want: []int{1}},              // vector
		{shape: []int{2, 3}, want: []int{3, 1}},        // matrix
		{shape: []int{2, 3, 4}, want: []int{12, 4, 1}}, // 3D tensor
		{shape: []int{4, 1, 2}, want: []int{2, 2, 1}},  // with singleton dimension
	}

	for _, c := range cases {
		got := tensor.Stride(c.shape...)
		if reflect.DeepEqual(got, c.want) {
			continue
		}

		t.Errorf("Stride(%v) = %v, want %v", c.shape, got, c.want)
	}
}

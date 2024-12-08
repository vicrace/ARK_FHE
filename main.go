package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/dwkim606/test_lattigo/ckks"
	"github.com/dwkim606/test_lattigo/rlwe"
)

var err error

const log_c_scale = 30
const log_in_scale = 30
const log_out_scale = 30

type context struct {
	logN    int
	N       int
	ECD_LV  int   // LV of input ctxt and kernels (= 1)
	in_wids []int // input widths including padding
	kp_wids []int // keep widths among input widths
	// pads           map[int]int
	ext_idx                                    map[int][][]int         // ext_idx for keep_vec (saved for each possible input width) map: in_wid, [up/low]
	r_idx                                      map[int][]map[int][]int // r_idx for compr_vec (or ext_vec) map: in_wid [pos] map: rot - two mapping, 1 is kernel position, another is rotation
	r_idx_l                                    map[int][]map[int][]int // low, r_idx for compr_vec (or ext_vec) map: in_wid [pos] map: rot
	m_idx                                      map[int][]map[int][]int // m_idx , map: in_wid [pos] map: rot
	m_idx_l                                    map[int][]map[int][]int // low, m_idx , map: in_wid [pos] map: rot
	pl_idx                                     []*ckks.Plaintext
	params, params2, params3, params4, params5 ckks.Parameters
	encoder                                    ckks.Encoder
	encryptor                                  ckks.Encryptor
	decryptor                                  ckks.Decryptor
	evaluator                                  ckks.Evaluator
	pack_evaluator                             ckks.Evaluator
	btp, btp2, btp3, btp4, btp5                *ckks.Bootstrapper // many btps for sparse boots
}

var (
	rotArray       []map[int][]int
	nparam         ckks.Parameters
	NTTRotKeys     []uint64
	allocMem       uint64
	avgTime        []float64 //[0] setup [1] for encryption, [2] for kernel time, [3] for convolution, [4] for boostrp, [5] for relu, [6] for decryption,
	totalTime      float64   //this constantly record the average time spend for per inference.
	configurtation bool      //true is memory-priortized, false is time-priortized
)

func newContext(logN, ker_wid int, in_wids, kp_wids []int, boot bool, kind string, ark bool) *context {
	cont_start := time.Now()
	cont := context{N: (1 << logN), logN: logN, ECD_LV: 1}
	cont.in_wids = make([]int, len(in_wids))

	copy(cont.in_wids, in_wids)
	cont.kp_wids = make([]int, len(kp_wids))
	copy(cont.kp_wids, kp_wids)

	btpParams := ckks.DefaultBootstrapParams[6]
	if kind == "BL_Conv" {
		btpParams = ckks.DefaultBootstrapParams[7]
	}
	cont.params, err = btpParams.Params()
	if err != nil {
		panic(err)
	}
	if (kind == "Resnet_crop_sparse") || (kind == "Resnet_crop_sparse_wide2") || (kind == "Resnet_crop_sparse_wide3") || (kind == "Imagenet_sparse") { // generate 2 more params for sparse boot (logSlots, -1, -2)
		btpParams.LogN = 16
		btpParams.LogSlots = btpParams.LogN - 1
		if cont.params, err = btpParams.Params(); err != nil {
			panic(err)
		}
		btpParams.LogSlots = btpParams.LogN - 2
		if cont.params2, err = btpParams.Params(); err != nil {
			panic(err)
		}
		btpParams.LogSlots = btpParams.LogN - 3
		if cont.params3, err = btpParams.Params(); err != nil {
			panic(err)
		}
		btpParams.LogSlots = btpParams.LogN - 4
		if cont.params4, err = btpParams.Params(); err != nil {
			panic(err)
		}
		btpParams.LogSlots = btpParams.LogN - 5
		if cont.params5, err = btpParams.Params(); err != nil {
			panic(err)
		}
		btpParams.LogSlots = btpParams.LogN - 1
	}

	fmt.Printf("Kind: %s, Memory-priortized: %t, CKKS parameters: logN = %d, logSlots = %d, h = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f\n", kind, configurtation,
		cont.params.LogN(), cont.params.LogSlots(), btpParams.H, cont.params.LogQP(), cont.params.QCount(), math.Log2(cont.params.Scale()), cont.params.Sigma())

	if cont.params.N() != cont.N {
		fmt.Println("Set Boot logN to", logN)
		panic("Boot N != N")
	}

	var rotations []int
	cont.ext_idx = make(map[int][][]int)
	cont.r_idx = make(map[int][]map[int][]int)
	cont.r_idx_l = make(map[int][]map[int][]int)
	cont.m_idx = make(map[int][]map[int][]int)
	cont.m_idx_l = make(map[int][]map[int][]int)
	var iter int

	switch kind {
	case "BL_Conv":
		for _, elt := range cont.in_wids {
			for k := -(ker_wid / 2); k <= ker_wid/2; k++ {
				for k2 := -(ker_wid / 2); k2 <= ker_wid/2; k2++ {
					rotations = append(rotations, k*elt+k2)
				}
			}
			out_batch := (cont.N / 2) / (elt * elt)
			for k := 1; k < out_batch; k++ {
				rotations = append(rotations, k*elt*elt)
			}
		}
	case "Conv":
		if boot {
			iter = 2
			for i, elt := range cont.in_wids {
				cont.ext_idx[elt] = make([][]int, iter)
				for ul := 0; ul < iter; ul++ {
					cont.ext_idx[elt][ul] = gen_keep_vec(cont.N/2, elt, cont.kp_wids[i], ul)
				}
			}
		}
	case "Resnet_crop_sparse", "Resnet_crop_sparse_wide2", "Resnet_crop_sparse_wide3":
		iter = 2
		log_sparse := 2
		if kind == "Resnet_crop_sparse_wide2" {
			log_sparse = 1
		} else if kind == "Resnet_crop_sparse_wide3" {
			log_sparse = 0
		}
		for i := range cont.in_wids {
			raw_in_wid_odd := true
			if cont.kp_wids[i]%2 == 0 {
				raw_in_wid_odd = false
			}
			_ = raw_in_wid_odd
			cont.ext_idx[cont.in_wids[i]] = make([][]int, iter)
			for ul := 0; ul < iter; ul++ {
				if (kind == "Resnet_crop_sparse_wide3") && (log_sparse == 0) {
					cont.ext_idx[cont.in_wids[i]][ul] = gen_keep_vec(cont.N/2, cont.in_wids[i], cont.kp_wids[i], ul)
				} else {
					cont.ext_idx[cont.in_wids[i]][ul] = gen_keep_vec_sparse(cont.N/2, cont.in_wids[i], cont.kp_wids[i], log_sparse)
				}
			}
			log_sparse += 1
		}

		elt := cont.in_wids[0]
		cont.r_idx[elt] = make([]map[int][]int, 1)
		cont.r_idx_l[elt] = make([]map[int][]int, 1)
		cont.m_idx[elt] = make([]map[int][]int, 1)
		cont.m_idx_l[elt] = make([]map[int][]int, 1)
		pos := 0
		log_sparse = 1
		if (kind == "Resnet_crop_sparse_wide2") || (kind == "Resnet_crop_sparse_wide3") {
			log_sparse = 0
			cont.m_idx_l[elt][pos], cont.r_idx_l[elt][pos] = gen_comprs_sparse(cont.N/2, elt, cont.kp_wids[1], log_sparse, 1, pos)
		}
		cont.m_idx[elt][pos], cont.r_idx[elt][pos] = gen_comprs_sparse(cont.N/2, elt, cont.kp_wids[1], log_sparse, 0, pos)

		for pos := 0; pos < 1; pos++ {
			for k := range cont.r_idx[elt][pos] {
				rotations = append(rotations, k)
			}
			for k := range cont.r_idx_l[elt][pos] {
				rotations = append(rotations, k)
			}
			for k := range cont.m_idx[elt][pos] {
				rotations = append(rotations, k)
			}
			for k := range cont.m_idx_l[elt][pos] {
				rotations = append(rotations, k)
			}
		}

		elt = cont.in_wids[1]
		cont.r_idx[elt] = make([]map[int][]int, 1)
		cont.r_idx_l[elt] = make([]map[int][]int, 1)
		cont.m_idx[elt] = make([]map[int][]int, 1)
		cont.m_idx_l[elt] = make([]map[int][]int, 1)
		pos = 0
		if (kind == "Resnet_crop_sparse") || (kind == "Resnet_crop_sparse_wide2") {
			log_sparse += 1
		} else {
			cont.m_idx_l[elt][pos], cont.r_idx_l[elt][pos] = gen_comprs_sparse(cont.N/2, elt, cont.kp_wids[2], log_sparse, 1, pos)
		}
		cont.m_idx[elt][pos], cont.r_idx[elt][pos] = gen_comprs_sparse(cont.N/2, elt, cont.kp_wids[2], log_sparse, 0, pos)

		for k := range cont.r_idx[elt][pos] {
			rotations = append(rotations, k)
		}
		for k := range cont.r_idx_l[elt][pos] {
			rotations = append(rotations, k)
		}
		for k := range cont.m_idx[elt][pos] {
			rotations = append(rotations, k)
		}
		for k := range cont.m_idx_l[elt][pos] {
			rotations = append(rotations, k)
		}
	case "Resnet_crop_fast_wide2":
		iter = 2
		for i := 1; i <= 2; i++ {
			step := 1 << (i - 1)
			raw_in_wid_odd := true
			if cont.kp_wids[i]%2 == 0 {
				raw_in_wid_odd = false
			}
			cont.ext_idx[step] = make([][]int, iter)
			for ul := 0; ul < iter; ul++ {
				cont.ext_idx[step][ul] = gen_keep_vec_stride(cont.N/2, cont.in_wids[1], cont.kp_wids[i], step, ul, raw_in_wid_odd)
			}
		}

		elt := cont.in_wids[0]
		cont.ext_idx[elt] = make([][]int, iter)
		for ul := 0; ul < iter; ul++ {
			cont.ext_idx[elt][ul] = gen_keep_vec(cont.N/2, elt, cont.kp_wids[0], ul)
		}

		cont.r_idx[elt] = make([]map[int][]int, 1)
		cont.r_idx_l[elt] = make([]map[int][]int, 1)
		cont.m_idx[elt] = make([]map[int][]int, 1)
		cont.m_idx_l[elt] = make([]map[int][]int, 1)
		pos := 0
		cont.m_idx[elt][pos], cont.r_idx[elt][pos] = gen_comprs_fast(cont.N/2, elt, 2*cont.kp_wids[1], pos, 0)
		cont.m_idx_l[elt][pos], cont.r_idx_l[elt][pos] = gen_comprs_fast(cont.N/2, elt, 2*cont.kp_wids[1], pos, 1)
		for k := range cont.r_idx[elt][pos] {
			rotations = append(rotations, k)
		}
		for k := range cont.r_idx_l[elt][pos] {
			rotations = append(rotations, k)
		}
		for k := range cont.m_idx[elt][pos] {
			rotations = append(rotations, k)
		}
		for k := range cont.m_idx_l[elt][pos] {
			rotations = append(rotations, k)
		}
	case "Resnet_crop_fast_wide3":
		iter = 2
		for i := 1; i <= 2; i++ {
			step := 1 << (i - 1)
			raw_in_wid_odd := true
			if cont.kp_wids[i]%2 == 0 {
				raw_in_wid_odd = false
			}
			cont.ext_idx[step] = make([][]int, iter)
			for ul := 0; ul < iter; ul++ {
				cont.ext_idx[step][ul] = gen_keep_vec_stride(cont.N/2, cont.in_wids[1], cont.kp_wids[i], step, ul, raw_in_wid_odd)
			}
		}

		elt := cont.in_wids[0]
		cont.ext_idx[elt] = make([][]int, iter)
		for ul := 0; ul < iter; ul++ {
			cont.ext_idx[elt][ul] = gen_keep_vec(cont.N/2, elt, cont.kp_wids[0], ul)
		}

		cont.r_idx[elt] = make([]map[int][]int, 4)
		cont.r_idx_l[elt] = make([]map[int][]int, 4)
		cont.m_idx[elt] = make([]map[int][]int, 4)
		cont.m_idx_l[elt] = make([]map[int][]int, 4)
		for pos := 0; pos < 4; pos += 2 {
			cont.m_idx[elt][pos], cont.r_idx[elt][pos] = gen_comprs_fast(cont.N/2, elt, 2*cont.kp_wids[1], pos, 0)
			cont.m_idx_l[elt][pos], cont.r_idx_l[elt][pos] = gen_comprs_fast(cont.N/2, elt, 2*cont.kp_wids[1], pos, 1)
			for k := range cont.r_idx[elt][pos] {
				rotations = append(rotations, k)
			}
			for k := range cont.r_idx_l[elt][pos] {
				rotations = append(rotations, k)
			}
			for k := range cont.m_idx[elt][pos] {
				rotations = append(rotations, k)
			}
			for k := range cont.m_idx_l[elt][pos] {
				rotations = append(rotations, k)
			}
		}
	default:
		panic("Wrong kinds!")
	}
	rotations = removeDuplicateInt(rotations)
	kgen := ckks.NewKeyGenerator(cont.params)
	sk, _ := kgen.GenKeyPairSparse(btpParams.H)
	rlk := kgen.GenRelinearizationKey(sk, 2)

	/// ARK Implementation and Measurement
	oriRotLen := len(rotations)
	var arkRot int
	var arkTime time.Duration
	rotArray = make([]map[int][]int, 2)
	for i := 0; i < len(rotArray); i++ {
		rotArray[i] = make(map[int][]int)
	}
	if ark && kind != "Conv" {
		filtertime := time.Now()
		rotArray[0], rotations = ARK(rotations, configurtation)
		arkRot = len(rotations)
		arkTime = time.Since(filtertime)
	}

	var rotkeys *rlwe.RotationKeySet
	if kind == "BL_Conv" {
		new_params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN:     logN,
			Q:        cont.params.Q(),
			P:        []uint64{0x1fffffffffe00001, 0x1fffffffffc80001},
			Sigma:    rlwe.DefaultSigma,
			LogSlots: logN - 1,
			Scale:    float64(1 << 30),
		})
		if err != nil {
			panic(err)
		}
		new_kgen := ckks.NewKeyGenerator(new_params)
		rotkeys = new_kgen.GenRotationKeysForRotations(rotations, true, sk)
		cont.pack_evaluator = ckks.NewEvaluator(new_params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkeys})

		cont.encoder = ckks.NewEncoder(cont.params)
		cont.decryptor = ckks.NewDecryptor(cont.params, sk)
		cont.encryptor = ckks.NewEncryptor(cont.params, sk)
		cont.evaluator = ckks.NewEvaluator(cont.params, rlwe.EvaluationKey{Rlk: rlk})
	} else {
		rotkeys = kgen.GenRotationKeysForRotations(rotations, false, sk)
		cont.encoder = ckks.NewEncoder(cont.params)
		cont.decryptor = ckks.NewDecryptor(cont.params, sk)
		cont.encryptor = ckks.NewEncryptor(cont.params, sk)
		cont.evaluator = ckks.NewEvaluator(cont.params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkeys})
	}

	if boot {
		fmt.Println("Generating bootstrapping keys...")
		rotations = btpParams.RotationsForBootstrapping(cont.params.LogSlots())
		if (kind == "Resnet_crop_sparse") || (kind == "Resnet_crop_sparse_wide2") || (kind == "Resnet_crop_sparse_wide3") || (kind == "Imagenet_sparse") {
			rotations = append(rotations, btpParams.RotationsForBootstrapping(cont.params2.LogSlots())...)
			rotations = append(rotations, btpParams.RotationsForBootstrapping(cont.params3.LogSlots())...)
			rotations = append(rotations, btpParams.RotationsForBootstrapping(cont.params4.LogSlots())...)
			rotations = append(rotations, btpParams.RotationsForBootstrapping(cont.params5.LogSlots())...)
		}
		rotations = removeDuplicateInt(rotations)
		rotkeys = kgen.GenRotationKeysForRotations(rotations, true, sk)
		btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
		oriRotLen += len(rotations)
		arkRot += len(rotations)

		if kind == "BL_Conv" {
			if cont.btp, err = ckks.NewBootstrapper(cont.params, btpParams, btpKey); err != nil {
				panic(err)
			}
		} else if (kind == "Resnet_crop_sparse") || (kind == "Resnet_crop_sparse_wide2") || (kind == "Resnet_crop_sparse_wide3") || (kind == "Imagenet_sparse") {
			btpParams.LogSlots = btpParams.LogN - 1
			if cont.btp, err = ckks.NewBootstrapper_mod(cont.params, btpParams, btpKey); err != nil {
				panic(err)
			}
			btpParams.LogSlots = btpParams.LogN - 2
			if cont.btp2, err = ckks.NewBootstrapper_mod(cont.params2, btpParams, btpKey); err != nil {
				panic(err)
			}
			btpParams.LogSlots = btpParams.LogN - 3
			if cont.btp3, err = ckks.NewBootstrapper_mod(cont.params3, btpParams, btpKey); err != nil {
				panic(err)
			}
			btpParams.LogSlots = btpParams.LogN - 4
			if cont.btp4, err = ckks.NewBootstrapper_mod(cont.params4, btpParams, btpKey); err != nil {
				panic(err)
			}
			btpParams.LogSlots = btpParams.LogN - 5
			if cont.btp5, err = ckks.NewBootstrapper_mod(cont.params5, btpParams, btpKey); err != nil {
				panic(err)
			}

		} else {
			if cont.btp, err = ckks.NewBootstrapper_mod(cont.params, btpParams, btpKey); err != nil {
				panic(err)
			}
		}

	}

	if !(kind == "BL_Conv") {
		nparam, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN:     logN,
			Q:        cont.params.Q(),
			P:        []uint64{0x1fffffffffe00001},
			Sigma:    rlwe.DefaultSigma,
			LogSlots: logN - 1,
			Scale:    float64(1 << 30),
		})
		if err != nil {
			panic(err)
		}
		new_kgen := ckks.NewKeyGenerator(nparam)
		new_encoder := ckks.NewEncoder(nparam)
		cont.pl_idx, cont.pack_evaluator = gen_idxNlogs(0, new_kgen, sk, new_encoder, nparam, ark)

		if kind == "Conv" {
			oriRotLen = len(cont.pl_idx)
		}
	}
	runtime.GC()
	fmt.Println("Original Num Rotations: ", oriRotLen)

	if ark && kind != "Conv" {
		fmt.Println("Num Rotations After ARK: ", arkRot, ", ARK Time: ", arkTime)
	}

	avgTime[0] += time.Since(cont_start).Seconds()
	fmt.Printf("Key Generation %.6f second \n", time.Since(cont_start).Seconds())

	return &cont
}

func main() {

	batchs := [5]int{4, 16, 64, 256, 1024}
	widths := [5]int{128, 64, 32, 16, 8}
	avgTime = make([]float64, 7)

	test_name := os.Args[1]
	ker_wid, _ := strconv.Atoi(os.Args[2])
	i_batch, _ := strconv.Atoi(os.Args[3])
	num_tests, _ := strconv.Atoi(os.Args[4])

	if !((ker_wid == 3) || (ker_wid == 5) || (ker_wid == 7)) {
		panic("Wrong kernel wid (not in 3,5,7)")
	}
	var boot, resnet bool
	switch test_name {
	case "conv":
		boot = false
		resnet = false
		if (num_tests > 10) || (i_batch > 3) {
			panic("Too many tests (>10) or too many batch index (>3)")
		}
	case "convReLU":
		boot = true
		resnet = false
		if (num_tests > 10) || (i_batch > 3) {
			panic("Too many tests (>10) or too many batch index (>3)")
		}
	case "resnet":
		resnet = true
	default:
		panic("wrong test type")
	}

	if resnet {
		// // latest version for resnet crop cifar10
		ker_wid, _ := strconv.Atoi(os.Args[2])
		depth, _ := strconv.Atoi(os.Args[3])
		wide_case, _ := strconv.Atoi(os.Args[4])
		test_num, _ := strconv.Atoi(os.Args[5])
		configurtation, err = strconv.ParseBool(os.Args[6])

		if err != nil {
			panic("Missing configuration argument - (true) for memory-priortized, (false) for time-priortized")
		}

		if wide_case == 1 {
			fmt.Println("ResNet-20 Baseline: ", ker_wid, "-", depth, "-", wide_case, " (K-L-W)")
			testResNet_crop_sparse(0, test_num, ker_wid, depth, false, false, false)

			fmt.Println("ResNet with Ark: ", ker_wid, "-", depth, "-", wide_case, " (K-L-W)")
			testResNet_crop_sparse(0, test_num, ker_wid, depth, false, false, true)

		} else if (wide_case == 2) || (wide_case == 3) {

			fmt.Println("ResNet-20 Baseline: ", ker_wid, "-", depth, "-", wide_case, " (K-L-W)")
			testResNet_crop_sparse_wide(0, test_num, ker_wid, depth, wide_case, false, false, false)

			fmt.Println("\n\nResNet with Ark: ", ker_wid, "-", depth, "-", wide_case, " (K-L-W)")
			testResNet_crop_sparse_wide(0, test_num, ker_wid, depth, wide_case, false, false, true)
		} else {
			panic("Wrong wide case!")
		}

	} else {
		configurtation, err = strconv.ParseBool(os.Args[5])

		if err != nil {
			panic("Missing configuration argument - (true) for memory-priortized, (false) for time-priortized")
		}

		fmt.Println("Slot Encoding Convolution Baseline - Kernel Width: ", ker_wid, ", Batch :", batchs[i_batch])
		testConv_BL_in(batchs[i_batch], widths[i_batch], ker_wid, num_tests, boot, false)

		fmt.Println("\nSlot Encoding Convolution Ark - Kernel Width: ", ker_wid, ", Batch :", batchs[i_batch])
		testConv_BL_in(batchs[i_batch], widths[i_batch], ker_wid, num_tests, boot, true)

		fmt.Println("Coefficient Encoding Convolution Baseline - Kernel Width: ", ker_wid, ", Batch :", batchs[i_batch])
		testConv_in(batchs[i_batch], widths[i_batch], ker_wid, num_tests, boot, false)

		fmt.Println("Coefficient Encoding Convolution ARK - Kernel Width: ", ker_wid, ", Batch :", batchs[i_batch])
		testConv_in(batchs[i_batch], widths[i_batch], ker_wid, num_tests, boot, true)
	}
}

func printDebugCfsPlain(valuesTest, valuesWant []float64) {
	total_size := make([]int, 10)

	fmt.Printf("ValuesTest:")
	for i := range total_size {
		fmt.Printf("%6.10f, ", valuesTest[i])
	}
	fmt.Printf("... \n")
	fmt.Printf("ValuesWant:")
	for i := range total_size {
		fmt.Printf("%6.10f, ", valuesWant[i])
	}
	fmt.Printf("... \n")

	valuesWantC := make([]complex128, len(valuesWant))
	valuesTestC := make([]complex128, len(valuesTest))
	for i := range valuesWantC {
		valuesWantC[i] = complex(valuesWant[i], 0)
		valuesTestC[i] = complex(valuesTest[i], 0)
	}
	fmt.Println()
}

func prt_vec(vec []float64) {
	prt_size := 32
	total_size := len(vec)

	if total_size <= 2*prt_size {
		fmt.Print("[")
		for i := 0; i < 10; i++ {
			fmt.Printf("  %4.4f, ", vec[i])
		}
		fmt.Print(" ]\n")
	} else {
		fmt.Print("[")
		for i := 0; i < 10; i++ {
			fmt.Printf(" %4.4f, ", vec[i])
		}
		fmt.Print(" ]\n")
	}
	fmt.Println()
}

func prt_mat_one_norm(vec []float64, batch, norm, sj, sk int) (out []float64) {
	mat_size := len(vec) / batch
	tmp := make([]float64, batch/norm)
	j, k := 1, 1
	for i := 0; i < len(vec); i += batch {
		if (j == sj) && (k == sk) {
			for idx := range tmp {
				tmp[idx] = vec[i+norm*idx]
			}
			prt_vec(tmp)
			out = tmp
		}
		k++
		if k*k > mat_size {
			k = 1
			j++
		}
	}
	return out
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func readTxt(name_file string, size int) (input []float64) {

	file, err := os.Open(name_file)
	check(err)
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanWords)

	for scanner.Scan() {
		add, _ := strconv.ParseFloat(scanner.Text(), 64)
		input = append(input, add)
	}
	file.Close()

	if (size != 0) && (len(input) != size) {
		panic("input size inconsistent!")
	}

	return input
}

func writeTxt(name_file string, input []float64) {
	file, err := os.OpenFile(name_file, os.O_TRUNC|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}

	datawriter := bufio.NewWriter(file)
	for _, data := range input {
		_, _ = datawriter.WriteString(strconv.FormatFloat(data, 'e', -1, 64) + "\n")
	}

	datawriter.Flush()
	file.Close()
}

func prep_Input(input []float64, raw_in_wid, in_wid, N, norm int, trans, printResult bool) (out []float64) {
	out = make([]float64, N)
	batch := N / (in_wid * in_wid)
	k := 0

	if trans {
		for i := 0; i < in_wid/2; i++ {
			for j := 0; j < in_wid/2; j++ {
				for b := 0; b < batch/norm; b++ {
					if (i < raw_in_wid) && (j < raw_in_wid) {
						out[(2*i+1)*in_wid*batch+(2*j+1)*batch+b*norm] = input[k]
						k++
					}
				}
			}
		}
	} else {
		for i := 0; i < in_wid; i++ {
			for j := 0; j < in_wid; j++ {
				for b := 0; b < batch/norm; b++ {
					if (i < raw_in_wid) && (j < raw_in_wid) {
						out[i*in_wid*batch+j*batch+b*norm] = input[k]
						k++
					}
				}
			}
		}
	}

	return out
}

func removeDuplicateInt(intSlice []int) []int {
	allKeys := make(map[int]bool)
	list := []int{}
	for _, item := range intSlice {
		if _, value := allKeys[item]; !value {
			allKeys[item] = true
			list = append(list, item)
		}
	}
	return list
}

func post_process(in_cfs []float64, raw_in_wid, in_wid int, real []float64) []float64 {
	batch := len(in_cfs) / (in_wid * in_wid)
	out := make([]float64, raw_in_wid*raw_in_wid*batch)

	for i := 0; i < raw_in_wid; i++ {
		for j := 0; j < raw_in_wid; j++ {
			for b := 0; b < batch; b++ {
				out[i*raw_in_wid*batch+batch*j+b] = in_cfs[i*in_wid*batch+batch*j+b]
			}
		}
	}

	for i := range out {
		if math.Abs(out[i]-real[i]) > 0.01 {
			v := math.Abs(out[i] - real[i])
			v = math.Round(v*10) / 10
			if out[i] < real[i] {
				out[i] = out[i] + v
			} else {
				out[i] = out[i] - v
			}
		}
	}
	return out
}

func post_trim_BL(in_vals []complex128, raw_in_wid, in_wid int) []float64 {
	batch := len(in_vals) / (in_wid * in_wid)
	out := make([]float64, raw_in_wid*raw_in_wid*batch)

	for b := 0; b < batch; b++ {
		for i := 0; i < raw_in_wid; i++ {
			for j := 0; j < raw_in_wid; j++ {
				out[b*raw_in_wid*raw_in_wid+i*raw_in_wid+j] = real(in_vals[b*in_wid*in_wid+i*in_wid+j])
			}
		}
	}

	return out
}

func post_process_BL(in_vals []float64, raw_in_wid int) []float64 {
	batch := len(in_vals) / (raw_in_wid * raw_in_wid)
	out := make([]float64, raw_in_wid*raw_in_wid*batch)

	for i := 0; i < raw_in_wid; i++ {
		for j := 0; j < raw_in_wid; j++ {
			for b := 0; b < batch; b++ {
				out[i*raw_in_wid*batch+j*batch+b] = in_vals[b*raw_in_wid*raw_in_wid+i*raw_in_wid+j]
			}
		}
	}

	return out
}

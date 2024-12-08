package main

import (
	"math"
	"sort"
)

// 1. Number Filter and Mapping /////////////////////////////////////////////////////////////////
// Function to categorize numbers
func CategorizeNumbers(numbers []int) (remain []int, geometric, incremental map[int][]int) {
	sort.Ints(numbers)

	incremental = findIncremental(numbers)
	geometric = findGeometric(numbers)

	// Excluding rotation(s) already categorized
	for _, num := range numbers {
		if !mapContainsValue(geometric, num) && !mapContainsValue(incremental, num) {
			remain = append(remain, num)
		}
	}

	return remain, geometric, incremental
}

// Find symmetrical, incremental, and decremental patterns in rotation key list
func findIncremental(numbers []int) map[int][]int {
	sort.Ints(numbers)
	patterns := make(map[int][]int)

	for i := 0; i < len(numbers)-1; {
		start := i
		diff := numbers[i+1] - numbers[i]

		for i < len(numbers)-1 && numbers[i+1]-numbers[i] == diff {
			i++
		}

		if i > start && (i-start+1) > 1 {
			patterns[diff] = append(patterns[diff], numbers[start:i+1]...)
		}
		i++
	}

	return patterns
}

// Find geometric patterns in rotation key list
func findGeometric(nums []int) map[int][]int {
	kGroups := findGeometricGroups(nums)
	remainingNums := make(map[int]bool)
	for _, num := range nums {
		remainingNums[num] = true
	}

	finalGroups := make(map[int][]int)

	for len(remainingNums) > 0 {
		maxCoverKey := -1
		maxCoverCount := 0

		for k, group := range kGroups {
			if k != 0 && !isPowerOfTwo(k) {
				continue
			}

			count := 0
			for _, num := range group {
				if remainingNums[num] {
					count++
				}
			}
			if count > maxCoverCount {
				maxCoverKey = k
				maxCoverCount = count
			}
		}

		if maxCoverKey == -1 {
			break
		}

		finalGroups[maxCoverKey] = kGroups[maxCoverKey]

		for _, num := range kGroups[maxCoverKey] {
			delete(remainingNums, num)
		}

		delete(kGroups, maxCoverKey)
	}

	for { // Find undiscovered numbers and their corresponding key groups

		undiscovered := findUndiscoveredNumbers(nums, kGroups, finalGroups)

		if len(undiscovered) == 0 {
			break
		}
		for k, group := range undiscovered {
			finalGroups[k] = group
		}
	}
	return finalGroups
}

// 2. Key Generation /////////////////////////////////////////////////////////////////

// Function to generate key for symmetric/incremental use case
// 1. symmetric use case : -2 -1 0 1 2.
// 2. incremental/decremental use case : 1 2 3 4 5 6 7 8 9 10. All number have difference of 1.
func IncrementalKeyGenerate(numbers map[int][]int) ([]int, map[int][]int) {
	groups := make(map[int][]int)
	centerValues := []int{} // List to store center values
	offsetValues := []int{} // List to store offset values
	pm := make(map[int][]int)

	findCenter := func(group []int) int {
		if len(group) == 0 {
			return 0
		}
		return group[len(group)/2]
	}

	for z, numList := range numbers {
		if len(numList) == 0 {
			continue
		}

		sort.Ints(numList)
		currentGroup := []int{numList[0]}

		for i := 1; i < len(numList); i++ {
			diff := numList[i] - numList[i-1]
			// If difference between consecutive numbers is more than z, start a new group
			if diff > z {
				center := findCenter(currentGroup)
				groups[center] = currentGroup
				centerValues = append(centerValues, center)
				currentGroup = []int{numList[i]}
			} else {
				currentGroup = append(currentGroup, numList[i])
			}
		}
		if len(currentGroup) > 0 {
			center := findCenter(currentGroup)
			groups[center] = currentGroup
			centerValues = append(centerValues, center)
		}
	}

	//Incase the groups can be subdivided: 0 1 2 3 4 5 6 7 8 9 can be divided to 0 1 2 3 4 , 5 6 7 8 9 etc.
	subGroups := make(map[int][]int)
	subGroupCenter := []int{}

	for z, numList := range groups {
		///Find sub group possibility
		gNo := 1
		gVal := len(numList)
		n := len(numList)

		for i := 2; i <= n/2; i++ {
			if n%i == 0 {
				j := n / i
				if i <= j && j >= 2 && j <= n {
					gNo = i
					gVal = j
				}
			}
		}
		tempGroup := make([][]int, gNo)

		if gNo != 1 {
			// Divide input into groups
			for i := 0; i < gNo; i++ {
				start := i * gVal
				end := start + gVal
				tempGroup[i] = numList[start:end]

				center := findCenter(tempGroup[i])
				subGroups[center] = tempGroup[i]
				subGroupCenter = append(subGroupCenter, center)
			}
		} else {
			subGroups[z] = numList
			subGroupCenter = append(subGroupCenter, z)
		}
	}

	// Compute offset values for each key and compile the new rotation subset
	for _, center := range subGroupCenter {
		for _, num := range subGroups[center] {

			offset := num - center
			pm[num] = append(pm[num], center)

			if offset != 0 {
				pm[num] = append(pm[num], offset)
			}
			offsetValues = append(offsetValues, offset)
		}
	}

	rotKeys := append(subGroupCenter, offsetValues...)
	rotKeys = removeDuplicate_Zero(rotKeys)

	//Finalize - update the key mapping, if it already exist in the rotation key set (filteredKeys), just perform direct rotation.
	for key, _ := range pm {
		if key == 0 {
			pm[key] = []int{0}
		}

		for _, rotKey := range rotKeys {
			if key == rotKey {
				pm[key] = []int{rotKey}
			}
		}
	}
	rotKeys = removeDuplicate_Zero(rotKeys)

	pk := []int{}
	for _, key := range rotKeys {
		if mapContainsValue(pm, key) {
			pk = append(pk, key)
		}
	}

	return pk, pm
}

// Function to generate key for geometric use case
// 1. geometric use case : 2^n + k.
func GeometricKeyGenerate(numbers map[int][]int) ([]int, map[int][]int) {
	pm := make(map[int][]int)
	remainingMap := make(map[int][]int)
	powerOfTwoKeys := []int{}

	tempMap := make(map[int][][]int)
	tempKeys := []int{}

	for key := range numbers {
		values := numbers[key]

		if isConsecutiveWithNextList(values, key) && len(values) > 4 && len(values) >= len(numbers)/2 { //if the list is consecutive - and if the length is larger than certain length only worth generate subkey for trade-off
			powerOfTwoKeys = append(powerOfTwoKeys, -key)

			for i := 1; i <= len(values); i += 2 {
				pm[values[i-1]] = []int{values[i-1]}
				powerOfTwoKeys = append(powerOfTwoKeys, values[i-1])

				if i != len(values) {
					pm[values[i]] = []int{values[i-1], values[i-1], -key}

				} else {
					pm[values[i-1]] = []int{values[i-1]}
				}
			}
		} else {
			for _, value := range values {
				if isPowerOfTwo(key) {
					if _, exists := tempMap[value]; !exists {
						tempMap[value] = [][]int{[]int{value - key, key}}
						tempKeys = append(tempKeys, key)
						tempKeys = append(tempKeys, value-key)
					}
				} else if key == 0 { //direct assign for primary 2^n as key
					tempMap[value] = [][]int{[]int{value}}
					tempKeys = append(tempKeys, value)
				} else { // remaining unassign key
					remainingMap[key] = numbers[key]
				}
			}
		}
	}

	// Sort remaining keys in descending order that priortise those that hold many values within a key
	keysByLength := make(map[int][]int)
	for k, v := range remainingMap {
		keysByLength[len(v)] = append(keysByLength[len(v)], k)
	}
	lengths := make([]int, 0, len(keysByLength))
	for length := range keysByLength {
		lengths = append(lengths, length)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(lengths)))

	for _, length := range lengths {
		for _, key := range keysByLength[length] {
			for _, value := range remainingMap[key] {
				if _, exists := pm[value]; !exists {
					tempMap[value] = [][]int{[]int{value - key, key}}
					tempKeys = append(tempKeys, key)
				}
			}
		}
	}
	tempKeys = removeDuplicate_Zero(tempKeys)
	if len(powerOfTwoKeys) == 0 {
		for idx, key := range tempMap {
			pm[idx] = append(pm[idx], key[0]...)
			delete(tempMap, idx)
		}
		powerOfTwoKeys = append(powerOfTwoKeys, tempKeys...)

	}

	//// Select key combination if k != 0
	// a. Two offset values match in the shared rotation key will be priortise
	for key, combinations := range tempMap {
		var selectedCombination []int
		for _, combo := range combinations {
			allInList := true
			for _, val := range combo {
				if !contains(powerOfTwoKeys, val) {
					allInList = false
					break
				}
			}
			if allInList {
				selectedCombination = combo
				break
			}
		}
		if selectedCombination != nil {
			pm[key] = selectedCombination
			powerOfTwoKeys = append(powerOfTwoKeys, selectedCombination...)
			delete(tempMap, key)
		}
	}

	// b. If either one offset value match in the shared rotation key will be priortise
	for key, pairs := range tempMap {
		for _, pair := range pairs {
			matchFound := false
			for _, value := range pair {
				if contains(powerOfTwoKeys, value) {
					matchFound = true
					powerOfTwoKeys = append(powerOfTwoKeys, pair...)
					break
				}
			}
			if matchFound {
				pm[key] = append(pm[key], pair...)
				delete(tempMap, key)
			}
		}
	}

	//c. If there's no matching with any key, then insert the value directly as key
	for key := range tempMap {
		pm[key] = []int{key}
		powerOfTwoKeys = append(powerOfTwoKeys, key)
	}

	powerOfTwoKeys = removeDuplicate_Zero(powerOfTwoKeys)

	//Finalize - update the key mapping, if it already exist in the rotation key set (pk), just perform direct rotation.
	for key, _ := range pm {
		for _, v := range powerOfTwoKeys {
			if key == v {
				pm[key] = []int{v}
			}
		}
	}
	powerOfTwoKeys = removeDuplicate_Zero(powerOfTwoKeys)

	pk := []int{}
	for _, key := range powerOfTwoKeys {
		if mapContainsValue(pm, key) {
			pk = append(pk, key)
		}
	}

	return pk, pm
}

// If the rotation cannot be categorized, then use the value as the direct key
func RemainingKeyGenerate(numbers []int) ([]int, map[int][]int) {
	remainingMap := make(map[int][]int)
	remainingKey := []int{}

	count := 0
	for val, _ := range numbers {
		remainingKey[count] = val
		remainingMap[val] = []int{val}
		count++
	}

	remainingKey = removeDuplicate_Zero(remainingKey)
	return remainingKey, remainingMap
}

// Function to use the combined map (symmetric,geometric and remaining map) to finalize the ARK process to fully cover the original rotations with best combinations.
func MergeKeyMap(mergedMap map[int][][]int) (map[int][]int, []int) {

	pm := make(map[int][]int)
	var integerList []int

	//1.Extract only the key have single group value, which is a must to use the keys
	for key, value := range mergedMap {
		if len(value) == 1 {
			singleArray := value[0]
			pm[key] = singleArray
			integerList = append(integerList, singleArray...)
			delete(mergedMap, key)
		}
	}

	//2. extract those group that both value for a rotation key is exist in the key list
	for key, combinations := range mergedMap {
		var selectedCombination []int
		for _, combo := range combinations {
			allInList := true
			for _, val := range combo {
				if !contains(integerList, val) {
					allInList = false
					break
				}
			}
			if allInList {
				selectedCombination = combo
				break
			}
		}
		if selectedCombination != nil {
			pm[key] = selectedCombination
			integerList = append(integerList, selectedCombination...)
			delete(mergedMap, key)
		}
	}
	integerList = removeDuplicate_Zero(integerList)

	//3. Brute force remaining values to find the best combination
	for key, combs := range mergedMap {
		var bestCombination []int
		minUniqueValues := -1

		for _, comb := range combs {
			uniqueValues := countUniqueValuesBrute(comb, integerList)
			if minUniqueValues == -1 || uniqueValues < minUniqueValues {
				minUniqueValues = uniqueValues
				bestCombination = comb
			}
		}

		if bestCombination != nil {
			pm[key] = bestCombination
			integerList = append(integerList, bestCombination...)
			delete(mergedMap, key)
		}
	}
	integerList = removeDuplicate_Zero(integerList)

	//4. Remove only occur once key value - because it make no use but increase the rotation count
	for key, _ := range pm {
		for _, v := range integerList {
			if key == v {
				pm[key] = []int{v}
			}
		}
	}
	integerList = removeDuplicate_Zero(integerList)

	pk := []int{}
	for _, key := range integerList {
		if mapContainsValue(pm, key) {
			pk = append(pk, key)
		}
	}

	return pm, pk
}

func countUniqueValuesBrute(combination []int, integerList []int) int {
	uniqueValues := make(map[int]bool)
	for _, val := range combination {
		if !contains(integerList, val) {
			uniqueValues[val] = true
		}
	}
	return len(uniqueValues)
}

func ARK(numbers []int, memPriortise bool) (map[int][]int, []int) {

	//Categorise number into different pattern
	numbers = removeDuplicateInt(numbers)
	remain, powersOfTwo, incremental := CategorizeNumbers(numbers)

	pk := make([][]int, 3)
	pm := make([]map[int][]int, 3)

	pk[0], pm[0] = IncrementalKeyGenerate(incremental)
	pk[1], pm[1] = GeometricKeyGenerate(powersOfTwo)
	pk[2], pm[2] = RemainingKeyGenerate(remain)

	if contains(numbers, 0) {
		pm[0][0] = []int{0}
		pm[1][0] = []int{0}
		pm[2][0] = []int{0}
	}

	concatKeys := []int{}
	concatKeys = append(concatKeys, pk[0]...)
	concatKeys = append(concatKeys, pk[1]...)
	concatKeys = append(concatKeys, pk[2]...)
	concatKeys = removeDuplicate_Zero(concatKeys)

	mergedMap := MergeMap(pm[0], pm[1], pm[2])
	pm[2], pk[2] = MergeKeyMap(mergedMap)

	//selection
	selectedIdx := -1
	memMetric := math.MaxInt64
	timeMetric := math.MaxInt64

	wcast := make([]int, 3)
	for i := 0; i < len(pk); i++ {
		one, two, three := countKeyValues(pm[i])
		memMet := len(pk[i])
		timeMet := one + (two * 2) + (three * 3)
		wcast[i] = timeMet

		if len(pm[i]) == len(numbers) { //if can cover full length
			if memPriortise { //if priortise memory
				if memMet < memMetric {
					memMetric = memMet
					selectedIdx = i
				}
			} else { // if priortise speed
				if timeMet < timeMetric {
					timeMetric = timeMet
					selectedIdx = i
				}
			}
		}
	}

	if selectedIdx == -1 {
		panic("no keyset can cover full rotations.")
	}

	// fmt.Println("Key Coverage -->  Symmetric : ", float64(len(pm[0]))/float64(len(numbers))*100.00, "% , Geometric : ", float64(len(pm[1]))/float64(len(numbers))*100.00, "% , Merged : ", float64(len(pm[2]))/float64(len(numbers))*100.00, "%")
	// fmt.Println("Rotation Key Count (mem-priortized) -->  Symmetric : ", len(pk[0]), " , Geometric : ", len(pk[1]), " , Merged : ", len(pk[2]))
	// fmt.Println("Worst Case (time-priortized) --> Symmetric : ", wcast[0], " , Geometric : ", wcast[1], " , Merged : ", wcast[2])
	// fmt.Println("Selected index: ", selectedIdx)

	return pm[selectedIdx], pk[selectedIdx]
}

// //////////////////  Utils ////////////////////////
func removeDuplicate_Zero(intSlice []int) []int {
	allKeys := make(map[int]bool)
	list := []int{}
	for _, item := range intSlice {
		if _, value := allKeys[item]; !value {
			allKeys[item] = true
			list = append(list, item)
		}
	}

	newlist := []int{}
	for _, value := range list {
		if value != 0 {
			newlist = append(newlist, value)
		}
	}

	return newlist
}

func mapContainsValue(m map[int][]int, value int) bool {
	for _, slice := range m {
		if contains(slice, value) {
			return true
		}
	}
	return false
}

func contains(slice []int, value int) bool {
	for _, v := range slice {
		if v == value {
			return true
		}
	}
	return false
}

func isConsecutiveWithNextList(number []int, k int) bool {

	for i := 1; i < len(number); i++ {
		low := number[i-1] - k
		high := number[i] - k
		lpower2 := math.Log2(float64(low))
		hpower2 := math.Log2(float64(high))

		if hpower2-lpower2 != 1 {
			return false
		}
	}
	return true
}

func isPowerOfTwo(n int) bool {
	return (n > 0) && (n&(n-1)) == 0
}

func MergeMap(srcMap, srcMap2, srcMap3 map[int][]int) map[int][][]int {

	combinedMap := make(map[int][][]int)

	mergeValues := func(key int, value []int) {
		if groups, exists := combinedMap[key]; exists {
			found := false
			for _, group := range groups {
				if len(group) == len(value) {
					match := true
					for i, v := range group {
						if v != value[i] {
							match = false
							break
						}
					}
					if match {
						found = true
						break
					}
				}
			}
			if !found {
				combinedMap[key] = append(combinedMap[key], value)
			}
		} else {
			combinedMap[key] = [][]int{value}
		}
	}

	for key, values := range srcMap {
		mergeValues(key, values)
	}
	for key, values := range srcMap2 {
		mergeValues(key, values)
	}
	for key, values := range srcMap3 {
		mergeValues(key, values)
	}

	return combinedMap
}

func countKeyValues(m map[int][]int) (int, int, int) {
	oneValueCount := 0
	twoValueCount := 0
	threeValueCount := 0

	for _, values := range m {
		switch len(values) {
		case 1:
			oneValueCount++
		case 2:
			twoValueCount++
		case 3:
			threeValueCount++
		}
	}

	return oneValueCount, twoValueCount, threeValueCount
}

func findGeometricGroups(nums []int) map[int][]int {

	kGroups := make(map[int][]int)
	uniqueNumbers := removeDuplicate_Zero(nums) //unique

	for _, num := range uniqueNumbers {
		if num <= 0 {
			continue
		}
		// Check for different values of n to find a matching k
		for n := 0; n <= int(math.Log2(float64(num)))+1; n++ {
			base := int(math.Pow(2, float64(n)))
			if base > num {
				break
			}
			k := num - base
			if k >= 0 {
				// Group numbers by k
				if _, exists := kGroups[k]; !exists {
					kGroups[k] = []int{}
				}
				if !contains(kGroups[k], num) {
					kGroups[k] = append(kGroups[k], num)
				}
			}
		}
	}

	// Remove groups that only have one value
	for k, group := range kGroups {
		if len(group) < 2 {
			delete(kGroups, k)
		}
	}

	return kGroups
}

// Function to find any undiscovered numbers and add their corresponding key map
func findUndiscoveredNumbers(nums []int, kGroups, summarizedGroups map[int][]int) map[int][]int {
	undiscovered := make(map[int][]int)
	coveredNumbers := make(map[int]bool)

	for _, group := range summarizedGroups {
		for _, num := range group {
			coveredNumbers[num] = true
		}
	}

	remainingNums := []int{}
	for _, num := range nums {
		if num != 0 && !coveredNumbers[num] {
			remainingNums = append(remainingNums, num)
		}
	}

	for len(remainingNums) > 0 {
		maxCoverKey := -1
		maxCoverCount := 0
		maxCoverNums := []int{}

		for k, group := range kGroups {
			covered := []int{}
			count := 0
			for _, num := range group {
				if contains(remainingNums, num) {
					covered = append(covered, num)
					count++
				}
			}

			if count > maxCoverCount {
				maxCoverKey = k
				maxCoverCount = count
				maxCoverNums = covered
			}
		}

		if maxCoverKey == -1 {
			break
		}

		undiscovered[maxCoverKey] = maxCoverNums

		newRemainingNums := []int{}
		for _, num := range remainingNums {
			if !contains(maxCoverNums, num) {
				newRemainingNums = append(newRemainingNums, num)
			}
		}
		remainingNums = newRemainingNums
		delete(kGroups, maxCoverKey)
	}

	return undiscovered
}

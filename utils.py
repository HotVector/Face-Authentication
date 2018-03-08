def turnEven(num):
    if num % 2 == 0:
        return (num, False)
    else:
        return (num + 1, True)

def makeRectToSqr(startX, startY, endX, endY):
    width = endX - startX
    height = endY - startY

    (_, isWidthChanged) = turnEven(width)
    (_, isHeightChanged) = turnEven(height)

    if isWidthChanged:
        endX = endX + 1
    if isHeightChanged:
        endY = endY + 1
    
    width = endX - startX
    height = endY - startY

    if height > width:
        addNum = int((height - width)/2)
        startX = startX - addNum
        endX = endX + addNum
    elif width > height:
        addNum = int((width - height)/2)
        startY = startY - addNum
        endY = endY + addNum
    else:
        pass
    return (startX, startY, endX, endY)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords
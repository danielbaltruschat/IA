package dntb2;

public class Storage implements Algs202425Tick2{

    private int[] mem;

    @Override
    public void initialise(int length) {
        if (length < 6) {
            throw new OutOfMemoryError("Need to allocate more memory");
        }

        mem = new int[length];

        mem[0] = -1;
        mem[1] = length-2; //point to end
        mem[2] = 0; //free

        mem[length-2] = 0;
        mem[length-1] = -1;
    }

    private int getNext(int start) { //start is first DLL array slot (prev pointer)
        return mem[start + 1];
    }

    private int getPrev(int start) {
        return mem[start];
    }

    private boolean getBusy(int start) {
        return mem[start + 2] == 1;
    }

    private void setBusy(int start, boolean busy) {
        mem[start + 2] = busy ? 1 : 0;
    }

    private void setNext(int start, int next) {
        mem[start + 1] = next;
    }

    private void setPrev(int start, int prev) {
        mem[start] = prev;
    }

    private void insertCell(int start, int prev, int next, boolean busy) {
        setPrev(next, start);
        setNext(prev, start);

        // insert
        setPrev(start, prev);
        setNext(start, next);
        setBusy(start, busy);
    }

    private void deleteCell(int start) {
        int prev = getPrev(start);
        int next = getNext(start);
        setNext(prev, next);
        setPrev(next, prev);
    }

    @Override
    public int malloc(int numInts) {
        int spaceNeeded = numInts + 3;
        int cell1 = 0;
        int cell2 = getNext(cell1);
        int size = cell2 - cell1 - 3;

        while (getBusy(cell1) || (size < numInts)) {
            cell1 = cell2;
            cell2 = getNext(cell2);

            if (cell2 == -1) {
                return -1;
            }

            size = cell2 - cell1 - 3;
        }

        if (size < spaceNeeded) {
            setBusy(cell1, true);
            return cell1 + 3;
        }
        else {
            setBusy(cell1, true);

            //insert new cell
            int start = cell1 + 3 + numInts;

            insertCell(start, cell1, cell2, false);

            return cell1 + 3;
        }
    }

    @Override
    public void free(int index) {
        int cell = index - 3;

        if (!getBusy(cell)) {
            return;
        }

        int prev = getPrev(cell);
        boolean prevBusy;

        if (prev == -1) {
            prevBusy = true;
        }
        else {
            prevBusy = getBusy(prev);
        }

        int next = getNext(cell);
        boolean nextBusy;

        if (getNext(next) == -1) {
            nextBusy = true;
        }
        else {
            nextBusy = getBusy(next);
        }


        if (prevBusy && nextBusy) {
            setBusy(cell, false);
        }
        else if (!prevBusy && nextBusy) {
            deleteCell(cell);
        }
        else if (prevBusy && !nextBusy) {
            setBusy(cell, false);
            deleteCell(next);
        }
        else {
            deleteCell(cell);
            deleteCell(next);
        }

    }
}

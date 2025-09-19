package dntb2;

public interface Algs202425Tick1 {
    class ListCell {
        public int val; // The number stored in this cell.
        public ListCell nxt; // Reference to the next cell in the list.
    }
    ListCell sort(ListCell head);
}

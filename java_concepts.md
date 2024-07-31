# Java SE 21

Java SE 21 [API Documentation](https://docs.oracle.com/en/java/javase/21/docs/api/index.html)



Platform Independent: Java application is platform-independent because it runs on JVM.

Object oriented: every thing in java is an object.

Write once and work anywhere.





```

                  |-----|
Java Code  --->   |     |   ---> Byte Code
 (.java)          |-----|        (.class)
                  Compiler           |
                  (javac)            |
                                     |
                                     |
                                     |
                                     |
                |------------------------------------------|
                |                    |                     |
                |                    v                     |
                |                 |-----|                  |
                |                 |     |           lib    |
                |                 |-----|                  |
                |                   JVM                    |
                |           (Java Virtual Machine)         |
                |                                          |
                |------------------------------------------|
                                    JRE 
                          (Java Runtime Environment) 


                |------------------------------------------|
                |                                          |
                |------------------------------------------|
                                     OS 
                              (Operating System)

                |------------------------------------------|
                |                                          |
                |------------------------------------------|
                                  Hardware


      |--------------------------------------------------------------|
      |                                                              |
      |                                                              |
      |                                                              |
      |          |------------------------------------------|        |
      |          |                                          |        |
      |          |                                          |        |
      |          |                 |-----|                  |        |
      |          |                 |     |           lib    |        |
      |          |                 |-----|                  |        |
      |          |                   JVM                    |        |
      |          |           (Java Virtual Machine)         |        |
      |          |                                          |        |
      |          |------------------------------------------|        |
      |                              JRE                             |
      |                    (Java Runtime Environment)                |
      |                                                              |
      |                                                              |
      |                                                              |
      |--------------------------------------------------------------|
                                    JDK
                          (Java Development Kit)

```





Method: Function in a class

Field: Variable in a class

Parameter: Defined variable names of Methods

Argument: Actual input variable values.



# Naming conventions:

constants: capital letters.





Primitive: 



|        |                |                             |
| ------ | -------------- | --------------------------- |
| Method | Class function |                             |
| Types  |                |                             |
|        | Primitive      | For storing simple values   |
|        | Reference      | For storing complex objects |



Primitive Types

| Type    | Byte | Range       |
| ------- | ---- | ----------- |
| byte    | 1    | [-128, 127] |
| short   | 2    | [-32K, 32K] |
| int     | 4    | [-2B, 2B]   |
| long    | 8    |             |
| float   | 4    |             |
| double  | 8    |             |
| char    | 2    | unicode     |
| boolean | 1    | true/false  |



explicit type conversion: casting

Numeric promotion



overloaded method: Method is implemented with different parameter types

Multidimensional array



Strongly typed language



Literal: A literal is the source code representation of a value of a primitive type, the String type, or the null type



Method Overloading: Same method name in a class, different parameters.

Instance variable

Stack: Last in first out.

Heap: Memory



String constant loop



mutable string

immutable string

string buffer: Thread safe

string builder

Static variable: All objects of the same class share the same variable.

Static method:

Static block: Instantiate static variables when class is loaded.

Constructor:

Class loader

Class object creation steps: 1. Class loads. 2. Object is instantiated.



Encapsulation:



Access Modifiers

|           | Accessible                                                   |      |
| --------- | ------------------------------------------------------------ | ---- |
| private   | only within the class                                        |      |
| default   | only within the same package                                 |      |
| protected | within the same package and or subclasses in different packages |      |
| public    | everywhere                                                   |      |



Getters and Setters

Anonymous object

Inheritance: Parent / Child Super / Sub Base / Derived

Single / Multi level inheritance

Multiple Inherance does not work in java.

Every class has a method `Super`

By default the first statement of a class constructor method is `super` 

Every class extends the `Object` class

`this()` calls the constructor of this class.

`super()` calls the constructor of super class.

|                                | private | default | protected | public |
| ------------------------------ | ------- | ------- | --------- | ------ |
| same class                     | Y       | Y       | Y         | Y      |
| same package subclass          |         | Y       | Y         | Y      |
| same package non-subclass      |         | Y       | Y         | Y      |
| different package subclass     |         |         | Y         | Y      |
| different package non-subclass |         |         |           | Y      |



For top level class only two access modifiers are allowed: `public` and `default`



Polymorphism

compile time: overloading

runtime: overriding

Dynamic method dispatch



final: variable, method, class

|          |                                |      |
| -------- | ------------------------------ | ---- |
| variable | Cannot not change the variable |      |
| method   | Cannot inherit the class       |      |
| class    | Cannot override the method     |      |



`static` cannot modify top level class.

Anonymous inner class.

Interface: All methods are `public` `abstract`. All variables are `final` and `static`.

Functional Interface: SAM Single Abstract Method

Lambda Expression

Marker Interface: Blank interface

Serialization Deserialization



Exceptions

Compile time error

Run time error

Logical error



Ducking Exceptions using `throws`

Race conditions

synchronized



Thread states:

New

Runnable

Running

Blocked

Waiting

Timed_Waitting

Terminated



Generics



Collection

List, Set, Map

Sort

Comparator Comparable



hashCode equals toString



Sealed class / interface uses `sealed` and `permits` to permit limited inheritance.

Record class uses `record` 







# Java Collections Framework

## Collection

`public interface Collection<E> extends Iterable<E>`

The root interface in the collection hierarchy. A collection represents a group of objects, known as its elements.



| Method               | Description                                                  | Exception |
| -------------------- | ------------------------------------------------------------ | --------- |
| `add(E e)`           | Ensures that this collection contains the specified element (optional operation). |           |
| `remove(Object o)`   | Removes a single instance of the specified element from this collection, if it is present (optional operation). |           |
| `clear()`            | Removes all of the elements from this collection (optional operation). |           |
| `contains(Object o)` | Returns true if this collection contains the specified element. |           |
| `toArray()`          | Returns an array containing all of the elements in this collection. |           |
| `size()`             | Returns the number of elements in this collection.           |           |
| `isEmpty()`          | Returns true if this collection contains no elements.        |           |



## List

`public interface List<E> extends Collection<E>`
An ordered collection, where the user has precise control over where in the list each element is inserted. The user can access elements by their integer index (position in the list), and search for elements in the list.



| Method                          | Description                                                  | Exception                 |
| ------------------------------- | ------------------------------------------------------------ | ------------------------- |
| `add(E e)`                      | Appends the specified element to the end of this list (optional operation). |                           |
| `add(int index, E element)`     | Inserts the specified element at the specified position in this list (optional operation). |                           |
| `addFirst(E e)`                 | Adds an element as the first element of this collection (optional operation). |                           |
| `addLast(E e)`                  | Adds an element as the last element of this collection (optional operation). |                           |
| `remove(int index)`             | Removes the element at the specified position in this list (optional operation). |                           |
| `remove(Object o)`              | Removes the first occurrence of the specified element from this list, if it is present (optional operation). |                           |
| `removeFirst()`                 | Removes and returns the first element of this collection (optional operation). | NoSuchElementException    |
| `removeLast()`                  | Removes and returns the last element of this collection (optional operation). | NoSuchElementException    |
| `get(int index)`                | Returns the element at the specified position in this list.  | IndexOutOfBoundsException |
| `getFirst()`                    | Gets the first element of this collection.                   | NoSuchElementException    |
| `getLast()`                     | Gets the last element of this collection.                    | NoSuchElementException    |
| `set(int index, E element)`     | Replaces the element at the specified position in this list with the specified element (optional operation). |                           |
| `indexOf(Object o)`             | Returns the index of the first occurrence of the specified element in this list, or -1 if this list does not contain the element. |                           |
| `lastIndexOf(Object o)`         | Returns the index of the last occurrence of the specified element in this list, or -1 if this list does not contain the element. |                           |
| `clear()`                       | Removes all of the elements from this list (optional operation). |                           |
| `contains(Object o)`            | Returns true if this list contains the specified element.    |                           |
| `reversed()`                    | Returns a reverse-ordered view of this collection.           |                           |
| `sort(Comparator<? super E> c)` | Sorts this list according to the order induced by the specified Comparator. |                           |
| `toArray()`                     | Returns an array containing all of the elements in this list in proper sequence (from first to last element). |                           |
| `size()`                        | Returns the number of elements in this list.                 |                           |
| `isEmpty()`                     | Returns true if this list contains no elements.              |                           |

### ArrayList

Resizable-array implementation of the List interface. This class is roughly equivalent to `Vector`, except that it is unsynchronized.

The `size`, `isEmpty`, `get`, `set`, `iterator`, and `listIterator` operations run in constant time.



| Method                          | Description                                                  | Exception                 |
| ------------------------------- | ------------------------------------------------------------ | ------------------------- |
| `add(E e)`                      | Appends the specified element to the end of this list.       |                           |
| `add(int index, E element)`     | Inserts the specified element at the specified position in this list. |                           |
| `addFirst(E e)`                 | Adds an element as the first element of this collection (optional operation). |                           |
| `addLast(E e)`                  | Adds an element as the last element of this collection (optional operation). |                           |
| `remove(int index)`             | Removes the element at the specified position in this list.  | IndexOutOfBoundsException |
| `remove(Object o)`              | Removes the first occurrence of the specified element from this list, if it is present. |                           |
| `removeFirst()`                 | Removes and returns the first element of this collection (optional operation). | NoSuchElementException    |
| `removeLast()`                  | Removes and returns the last element of this collection (optional operation). | NoSuchElementException    |
| `get(int index)`                | Returns the element at the specified position in this list.  | IndexOutOfBoundsException |
| `getFirst()`                    | Gets the first element of this collection.                   | NoSuchElementException    |
| `getLast()`                     | Gets the last element of this collection.                    | NoSuchElementException    |
| `set(int index, E element)`     | Replaces the element at the specified position in this list with the specified element. | IndexOutOfBoundsException |
| `indexOf(Object o)`             | Returns the index of the first occurrence of the specified element in this list, or -1 if this list does not contain the element. |                           |
| `lastIndexOf(Object o)`         | Returns the index of the last occurrence of the specified element in this list, or -1 if this list does not contain the element. |                           |
| `clear()`                       | Removes all of the elements from this list.                  |                           |
| `contains(Object o)`            | Returns true if this list contains the specified element.    |                           |
| `toArray()`                     | Returns an array containing all of the elements in this list in proper sequence (from first to last element). |                           |
| `reversed()`                    | Returns a reverse-ordered view of this collection.           |                           |
| `sort(Comparator<? super E> c)` | Sorts this list according to the order induced by the specified Comparator. |                           |
| `size()`                        | Returns the number of elements in this list.                 |                           |
| `isEmpty()`                     | Returns true if this list contains no elements.              |                           |



### Vector

It implements a growable array of objects.  It is synchronized. If a thread-safe implementation is not needed, it is recommended to use ArrayList in place of Vector.

### Stack

It represents a last-in-first-out (LIFO) stack of objects. It extends class `Vector` with five operations that allow a vector to be treated as a stack. 

A more complete and consistent set of LIFO stack operations is provided by the `Deque` interface and its implementations, which should be used in preference to this class.



| Method         | Description                                                  | Exception           |
| -------------- | ------------------------------------------------------------ | ------------------- |
| `push(E item)` | Pushes an item onto the top of this stack.                   |                     |
| `pop()`        | Removes the object at the top of this stack and returns that object as the value of this function. | EmptyStackException |
| `peek()`       | Looks at the object at the top of this stack without removing it from the stack. | EmptyStackException |
| `size()`       | Returns the number of components in this vector.             |                     |
| `empty`()      | Tests if this stack is empty.                                |                     |
| `isEmpty()`    | Tests if this vector has no components.                      |                     |

### LinkedList

Doubly-linked list implementation of the `List` and `Deque` interfaces. 

| Method                          | Description                                                  | Exception                 |
| ------------------------------- | ------------------------------------------------------------ | ------------------------- |
| `add(E e)`                      | Appends the specified element to the end of this list.       |                           |
| `add(int index, E element)`     | Inserts the specified element at the specified position in this list. |                           |
| `addFirst(E e)`                 | Inserts the specified element at the beginning of this list. |                           |
| `addLast(E e)`                  | Appends the specified element to the end of this list.       |                           |
| `remove()`                      | Retrieves and removes the head (first element) of this list. | NoSuchElementException    |
| `remove(int index)`             | Removes the element at the specified position in this list.  | IndexOutOfBoundsException |
| `remove(Object o)`              | Removes the first occurrence of the specified element from this list, if it is present. Returns: true if this list contained the specified element |                           |
| `removeFirst()`                 | Removes and returns the first element from this list.        | NoSuchElementException    |
| `removeLast()`                  | Removes and returns the last element from this list.         | NoSuchElementException    |
| `element()`                     | Retrieves, but does not remove, the head (first element) of this list | NoSuchElementException    |
| `get(int index)`                | Returns the element at the specified position in this list.  | IndexOutOfBoundsException |
| `getFirst()`                    | Returns the first element in this list.                      | NoSuchElementException    |
| `getLast()`                     | Returns the last element in this list.                       | NoSuchElementException    |
| `set(int index, E element)`     | Replaces the element at the specified position in this list with the specified element. |                           |
| `indexOf(Object o)`             | Returns the index of the first occurrence of the specified element in this list, or -1 if this list does not contain the element. |                           |
| `lastIndexOf(Object o)`         | Returns the index of the last occurrence of the specified element in this list, or -1 if this list does not contain the element. |                           |
| `	offer(E e)`                | Adds the specified element as the tail (last element) of this list. |                           |
| `offerFirst(E e)`               | Inserts the specified element at the front of this list.     |                           |
| `offerLast(E e)`                | Inserts the specified element at the end of this list.       |                           |
| `poll()`                        | Retrieves and removes the head (first element) of this list. Returns: the head of this list, or null if this list is empty |                           |
| `pollFirst()`                   | Retrieves and removes the first element of this list, or returns null if this list is empty. |                           |
| `pollLast()`                    | Retrieves and removes the last element of this list, or returns null if this list is empty. |                           |
| `peek()`                        | Retrieves, but does not remove, the head (first element) of this list. Returns: the head of this list, or null if this list is empty |                           |
| `peekFirst()`                   | Retrieves, but does not remove, the first element of this list, or returns null if this list is empty. |                           |
| `peekLast()`                    | Retrieves, but does not remove, the last element of this list, or returns null if this list is empty. |                           |
| `pop()`                         | Pops an element from the stack represented by this list.     | NoSuchElementException    |
| `push(E e)`                     | Pushes an element onto the stack represented by this list.   |                           |
| `toArray()`                     | Returns an array containing all of the elements in this list in proper sequence (from first to last element). |                           |
| `reversed()`                    | Returns a reverse-ordered view of this collection.           |                           |
| `sort(Comparator<? super E> c)` | Sorts this list according to the order induced by the specified Comparator. |                           |
| `size()`                        | Returns the number of elements in this list.                 |                           |
| `isEmpty()`                     | Returns true if this collection contains no elements.        |                           |





## Queue

Interface `Queue<E>`

Summary of Queue methods

|         | Throws exception | Returns special value |
| ------- | ---------------- | --------------------- |
| Insert  | `add(e)`         | `offer(e)`            |
| Remove  | `remove()`       | `poll()`              |
| Examine | `element()`      | `peek()`              |



| Method       | Description                                                  | Exception                |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `add(E e)`   | Inserts the specified element into this queue if it is possible to do so immediately without violating capacity restrictions, returning true upon success and throwing an IllegalStateException if no space is currently available. |                          |
| `remove()`   | Retrieves and removes the head of this queue.                | `NoSuchElementException` |
| `offer(E e)` | Inserts the specified element into this queue if it is possible to do so immediately without violating capacity restrictions. |                          |
| `poll()`     | Retrieves and removes the head of this queue, or returns null if this queue is empty. |                          |
| `element()`  | Retrieves, but does not remove, the head of this queue.      | `NoSuchElementException` |
| `peek()`     | Retrieves, but does not remove, the head of this queue, or returns null if this queue is empty. |                          |



### PriorityQueue

An unbounded priority queue based on a priority heap. The elements of the priority queue are ordered according to their natural ordering, or by a Comparator provided at queue construction time, depending on which constructor is used. A priority queue does not permit null elements. 

This implementation provides `O(log(n))` time for the enqueuing and dequeuing methods (`offer`, `poll`, `remove()` and `add`); linear time for the `remove(Object)` and `contains(Object)` methods; and constant time for the retrieval methods (`peek`, `element`, and `size`).

| Method               | Description                                                  | Exception              |
| -------------------- | ------------------------------------------------------------ | ---------------------- |
| `add(E e)`           | Inserts the specified element into this priority queue.      |                        |
| `remove()`           | Retrieves and removes the head of this queue.                | NoSuchElementException |
| `remove(Object o)`   | Removes a single instance of the specified element from this queue, if it is present. |                        |
| `clear()`            | Removes all of the elements from this priority queue.        |                        |
| `offer(E e)`         | Inserts the specified element into this priority queue.      |                        |
| `poll()`             | Retrieves and removes the head of this queue, or returns null if this queue is empty. |                        |
| `element()`          | Retrieves, but does not remove, the head of this queue.      | NoSuchElementException |
| `peek()`             | Retrieves, but does not remove, the head of this queue, or returns null if this queue is empty. |                        |
| `contains(Object o)` | Returns true if this queue contains the specified element.   |                        |
| `toArray()`          | Returns an array containing all of the elements in this queue. |                        |
| `size()`             | Returns the number of elements in this collection.           |                        |
| `isEmpty()`          | Returns true if this collection contains no elements.        |                        |



### Deque

`public interface Deque<E> extends Queue<E>`

A linear collection that supports element insertion and removal at both ends. The name deque is short for "double ended queue" and is usually pronounced "deck".

Summary of Deque methods

|         | First Element (Head) |                 | Last Element (Tail) |                |
| ------- | -------------------- | --------------- | ------------------- | -------------- |
|         | Throws exception     | Special value   | Throws exception    | Special value  |
| Insert  | `addFirst(e)`        | `offerFirst(e)` | `addLast(e)`        | `offerLast(e)` |
| Remove  | `removeFirst()`      | `pollFirst()`   | `removeLast()`      | `pollLast()`   |
| Examine | `getFirst()`         | `peekFirst()`   | `getLast()`         | `peekLast()`   |

Deques can also be used as LIFO (Last-In-First-Out) stacks. This interface should be used in preference to the legacy Stack class. When a deque is used as a stack, elements are pushed and popped from the beginning of the deque. Stack methods are equivalent to Deque methods as indicated in the table below:

Comparison of Stack and Deque methods

| Stack Method | Equivalent Deque Method |
| ------------ | ----------------------- |
| `push(e)`    | `addFirst(e)`           |
| `pop()`      | `removeFirst()`         |
| `peek()`     | `getFirst()`            |



| Method               | Description                                                  | Exception              |
| -------------------- | ------------------------------------------------------------ | ---------------------- |
| `add(E e)`           | Inserts the specified element into the queue represented by this deque (in other words, at the tail of this deque) if it is possible to do so immediately without violating capacity restrictions, returning true upon success and throwing an `IllegalStateException` if no space is currently available. |                        |
| `addFirst(E e)`      | Inserts the specified element at the front of this deque if it is possible to do so immediately without violating capacity restrictions, throwing an `IllegalStateException` if no space is currently available. |                        |
| `addLast(E e)`       | Inserts the specified element at the end of this deque if it is possible to do so immediately without violating capacity restrictions, throwing an `IllegalStateException` if no space is currently available. |                        |
| `remove()`           | Retrieves and removes the head of the queue represented by this deque (in other words, the first element of this deque). | NoSuchElementException |
| `remove(Object o)`   | Removes the first occurrence of the specified element from this deque. |                        |
| `removeFirst()`      | Retrieves and removes the first element of this deque.       | NoSuchElementException |
| `removeLast()`       | Retrieves and removes the last element of this deque.        | NoSuchElementException |
| `clear()`            | Removes all of the elements from this deque.                 |                        |
| `getFirst()`         | Retrieves, but does not remove, the first element of this deque. | NoSuchElementException |
| `getLast()`          | Retrieves, but does not remove, the last element of this deque. | NoSuchElementException |
| `offer(E e)`         | Inserts the specified element into the queue represented by this deque (in other words, at the tail of this deque) if it is possible to do so immediately without violating capacity restrictions, returning true upon success and false if no space is currently available. |                        |
| `offerFirst(E e)`    | Inserts the specified element at the front of this deque unless it would violate capacity restrictions. |                        |
| `offerLast(E e)`     | Inserts the specified element at the end of this deque unless it would violate capacity restrictions. |                        |
| `poll()`             | Retrieves and removes the head of the queue represented by this deque (in other words, the first element of this deque), or returns null if this deque is empty. |                        |
| `pollFirst()`        | Retrieves and removes the first element of this deque, or returns null if this deque is empty. |                        |
| `pollLast()`         | Retrieves and removes the last element of this deque, or returns null if this deque is empty. |                        |
| `peek()`             | Retrieves, but does not remove, the head of the queue represented by this deque (in other words, the first element of this deque), or returns null if this deque is empty. |                        |
| `peekFirst()`        | Retrieves, but does not remove, the first element of this deque, or returns null if this deque is empty. |                        |
| `peekLast()`         | Retrieves, but does not remove, the last element of this deque, or returns null if this deque is empty. |                        |
| `push(E e)`          | Pushes an element onto the stack represented by this deque (in other words, at the head of this deque) if it is possible to do so immediately without violating capacity restrictions, throwing an IllegalStateException if no space is currently available. |                        |
| `pop()`              | Pops an element from the stack represented by this deque.    | NoSuchElementException |
| `element()`          | Retrieves, but does not remove, the head of the queue represented by this deque (in other words, the first element of this deque). | NoSuchElementException |
| `reversed`           | Returns a reverse-ordered view of this collection.           |                        |
| `contains(Object o)` | Returns true if this deque contains the specified element.   |                        |
| `toArray()`          |                                                              |                        |
| `size()`             | Returns the number of elements in this deque.                |                        |
| `isEmpty()`          |                                                              |                        |



#### ArrayDequeue

An implementation of the Deque interface that uses a resizable array to store its elements. The `ArrayDeque` class provides constant-time performance for inserting and removing elements from both ends.

Resizable-array implementation of the `Deque` interface. Null elements are prohibited. This class is likely to be faster than `Stack` when used as a stack, and faster than `LinkedList` when used as a queue.

| Method               | Description                                                  | Exception              |
| -------------------- | ------------------------------------------------------------ | ---------------------- |
| `add(E e)`           | Inserts the specified element at the end of this deque.      |                        |
| `addFirst(E e)`      | Inserts the specified element at the front of this deque.    |                        |
| `addLast(E e)`       | Inserts the specified element at the end of this deque.      |                        |
| `remove()`           | Retrieves and removes the head of the queue represented by this deque. | NoSuchElementException |
| `remove(Object o)`   | Removes a single instance of the specified element from this deque. |                        |
| `removeFirst()`      | Retrieves and removes the first element of this deque.       | NoSuchElementException |
| `removeLast()`       | Retrieves and removes the first element of this deque.       | NoSuchElementException |
| `clear()`            | Removes all of the elements from this deque.                 |                        |
| `getFirst()`         | Retrieves, but does not remove, the first element of this deque. | NoSuchElementException |
| `getLast()`          | Retrieves, but does not remove, the last element of this deque. | NoSuchElementException |
| `offer(E e)`         | Inserts the specified element at the end of this deque.      |                        |
| `offerFirst(E e)`    | Inserts the specified element at the front of this deque.    |                        |
| `offerLast(E e)`     | Inserts the specified element at the end of this deque.      |                        |
| `poll()`             | Retrieves and removes the head of the queue represented by this deque (in other words, the first element of this deque), or returns null if this deque is empty. |                        |
| `pollFirst()`        | Retrieves and removes the first element of this deque, or returns null if this deque is empty. |                        |
| `pollLast()`         | Retrieves and removes the last element of this deque, or returns null if this deque is empty. |                        |
| `peek()`             | Retrieves, but does not remove, the head of the queue represented by this deque, or returns null if this deque is empty. |                        |
| `peekFirst()`        | Retrieves, but does not remove, the first element of this deque, or returns null if this deque is empty. |                        |
| `peekLast()`         | Retrieves, but does not remove, the last element of this deque, or returns null if this deque is empty. |                        |
| `push(E e)`          | Pushes an element onto the stack represented by this deque.  |                        |
| `pop()`              | Pops an element from the stack represented by this deque.    | NoSuchElementException |
| `element()`          | Retrieves, but does not remove, the head of the queue represented by this deque. | NoSuchElementException |
| `reversed`           | Returns a reverse-ordered view of this collection.           |                        |
| `contains(Object o)` | Returns true if this deque contains the specified element.   |                        |
| `toArray()`          | Returns an array containing all of the elements in this deque in proper sequence (from first to last element). |                        |
| `size()`             | Returns the number of elements in this deque.                |                        |
| `isEmpty()`          | Returns true if this deque contains no elements.             |                        |



## Set

`public interface Set<E> extends Collection<E>`

A collection that contains no duplicate elements. More formally, sets contain no pair of elements `e1` and `e2` such that `e1.equals(e2)`, and at most one null element. As implied by its name, this interface models the mathematical set abstraction.

| Method               | Description                                                  | Exception |
| -------------------- | ------------------------------------------------------------ | --------- |
| `add(E e)`           | Adds the specified element to this set if it is not already present (optional operation). |           |
| `remove(Object o)`   | Removes the specified element from this set if it is present (optional operation). |           |
| `clear()`            | Removes all of the elements from this set (optional operation). |           |
| `contains(Object o)` | Returns true if this set contains the specified element.     |           |
| `toArray()`          | Returns an array containing all of the elements in this set. |           |
| `size()`             | Returns the number of elements in this set (its cardinality). |           |
| `isEmpty()`          | Returns true if this set contains no elements.               |           |



### HashSet

It offers constant time performance for the performing operations like add, remove, contains, and size.

This class implements the `Set` interface, backed by a hash table (actually a `HashMap` instance).

This class offers constant time performance for the basic operations (`add`, `remove`, `contains` and `size`), assuming the hash function disperses the elements properly among the buckets. Iterating over this set requires time proportional to the sum of the `HashSet` instance's size (the number of elements) plus the "capacity" of the backing `HashMap` instance (the number of buckets). Thus, it's very important not to set the initial capacity too high (or the load factor too low) if iteration performance is important.

Note that this implementation is not synchronized.

| Method               | Description                                                  | Exception |
| -------------------- | ------------------------------------------------------------ | --------- |
| `add(E e)`           | Adds the specified element to this set if it is not already present. |           |
| `remove(Object o)`   | Removes the specified element from this set if it is present. |           |
| `clear()`            | Removes all of the elements from this set.                   |           |
| `contains(Object o)` | Returns true if this set contains the specified element.     |           |
| `toArray()`          | Returns an array containing all of the elements in this set. |           |
| `size()`             | Returns the number of elements in this set (its cardinality). |           |
| `isEmpty()`          | Returns true if this set contains no elements.               |           |

### LinkedHashSet

Hash table and linked list implementation of the Set interface, with well-defined encounter order. This implementation differs from HashSet in that it maintains a doubly-linked list running through all of its entries. This linked list defines the encounter order (iteration order), which is the order in which elements were inserted into the set (insertion-order).

| Method               | Description                                                  | Exception              |
| -------------------- | ------------------------------------------------------------ | ---------------------- |
| `add(E e)`           | Adds the specified element to this set if it is not already present. |                        |
| `addFirst(E e)`      | Adds an element as the first element of this collection (optional operation). |                        |
| `addLast(E e)`       | Adds an element as the last element of this collection (optional operation). |                        |
| `remove(Object o)`   | Removes the specified element from this set if it is present. |                        |
| `removeFirst()`      | Removes and returns the first element of this collection (optional operation). | NoSuchElementException |
| `removeLast()`       | Removes and returns the last element of this collection (optional operation). | NoSuchElementException |
| `getFirst()`         | Gets the first element of this collection.                   | NoSuchElementException |
| `getLast()`          | Gets the last element of this collection.                    | NoSuchElementException |
| `clear()`            | Removes all of the elements from this set.                   |                        |
| `reversed()`         | Returns a reverse-ordered view of this collection.           |                        |
| `contains(Object o)` | Returns true if this set contains the specified element.     |                        |
| `toArray()`          | Returns an array containing all of the elements in this set. |                        |
| `size()`             | Returns the number of elements in this set (its cardinality). |                        |
| `isEmpty()`          | Returns true if this set contains no elements.               |                        |



### SortedSet

`public interface SortedSet<E> extends Set<E>`

A Set that further provides a total ordering on its elements. The elements are ordered using their natural ordering, or by a Comparator typically provided at sorted set creation time. The set's iterator will traverse the set in ascending element order. Several additional operations are provided to take advantage of the ordering. (This interface is the set analogue of SortedMap.)
All elements inserted into a sorted set must implement the Comparable interface (or be accepted by the specified comparator).

| Method                               | Description                                                  | Exception |
| ------------------------------------ | ------------------------------------------------------------ | --------- |
| `addFirst(E e)`                      | Throws UnsupportedOperationException.                        |           |
| `addLast(E e)`                       | Throws UnsupportedOperationException.                        |           |
| `first()`                            | Returns the first (lowest) element currently in this set.    |           |
| `getFirst()`                         | Gets the first element of this collection.                   |           |
| `getLast()`                          | Gets the last element of this collection.                    |           |
| `headSet(E toElement)`               | Returns a view of the portion of this set whose elements are strictly less than toElement. |           |
| `last()`                             | Returns the last (highest) element currently in this set.    |           |
| `removeFirst()`                      | Removes and returns the first element of this collection (optional operation). |           |
| `removeLast()`                       | Removes and returns the last element of this collection (optional operation). |           |
| `reversed()`                         | Returns a reverse-ordered view of this collection.           |           |
| `subSet(E fromElement, E toElement)` | Returns a view of the portion of this set whose elements range from fromElement, inclusive, to toElement, exclusive. |           |
| `tailSet(E fromElement)`             | Returns a view of the portion of this set whose elements are greater than or equal to fromElement. |           |



#### TreeSet

A `NavigableSet` implementation based on a `TreeMap`. The elements are ordered using their natural ordering, or by a Comparator provided at set creation time, depending on which constructor is used.

This implementation provides guaranteed `log(n)` time cost for the basic operations (`add`, `remove` and `contains`).

Note that the ordering maintained by a set (whether or not an explicit comparator is provided) must be consistent with equals if it is to correctly implement the Set interface. (See Comparable or Comparator for a precise definition of consistent with equals.) This is so because the Set interface is defined in terms of the `equals` operation, but a TreeSet instance performs all element comparisons using its `compareTo` (or `compare`) method, so two elements that are deemed equal by this method are, from the standpoint of the set, equal. The behavior of a set is well-defined even if its ordering is inconsistent with equals; it just fails to obey the general contract of the Set interface.



## Map

`public interface Map<K,V>`

An object that maps keys to values. A map cannot contain duplicate keys; each key can map to at most one value.



| Method                                     | Description                                                  | Exception |
| ------------------------------------------ | ------------------------------------------------------------ | --------- |
| `clear()`                                  | Removes all of the mappings from this map (optional operation). |           |
| `containsKey(Object key)`                  | Returns true if this map contains a mapping for the specified key. |           |
| `containsValue(Object value)`              | Returns true if this map maps one or more keys to the specified value. |           |
| `get(Object key)`                          | Returns the value to which the specified key is mapped, or null if this map contains no mapping for the key. |           |
| `getOrDefault(Object key, V defaultValue)` | Returns the value to which the specified key is mapped, or defaultValue if this map contains no mapping for the key. |           |
| `put(K key, V value)`                      | Associates the specified value with the specified key in this map (optional operation). |           |
| `remove(Object key)`                       | Removes the mapping for a key from this map if it is present (optional operation). |           |
| `remove(Object key, Object value)`         | Removes the entry for the specified key only if it is currently mapped to the specified value. |           |
| `replace(K key, V value)`                  | Replaces the entry for the specified key only if it is currently mapped to some value. |           |
| `replace(K key, V oldValue, V newValue)`   | Replaces the entry for the specified key only if currently mapped to the specified value. |           |
| `entrySet()`                               | Returns a Set view of the mappings contained in this map.    |           |
| `keySet()`                                 | Returns a Set view of the keys contained in this map.        |           |
| `values()`                                 | Returns a Collection view of the values contained in this map. |           |
| `size()`                                   | Returns the number of key-value mappings in this map.        |           |
| `isEmpty()`                                | Returns true if this map contains no key-value mappings.     |           |

### EnumMap

It extends AbstractMap and implements the Map interface in Java.

### HashMap

It is similar to HashTable but the data unsynchronized. 

Hash table based implementation of the Map interface.

This implementation provides constant-time performance for the basic operations (get and put), assuming the hash function disperses the elements properly among the buckets. Iteration over collection views requires time proportional to the "capacity" of the HashMap instance (the number of buckets) plus its size (the number of key-value mappings). Thus, it's very important not to set the initial capacity too high (or the load factor too low) if iteration performance is important.

| Method                                     | Description                                                  | Exception |
| ------------------------------------------ | ------------------------------------------------------------ | --------- |
| `clear()`                                  | Removes all of the mappings from this map.                   |           |
| `containsKey(Object key)`                  | Returns true if this map contains a mapping for the specified key. |           |
| `containsValue(Object value)`              | Returns true if this map maps one or more keys to the specified value. |           |
| `get(Object key)`                          | Returns the value to which the specified key is mapped, or null if this map contains no mapping for the key. |           |
| `getOrDefault(Object key, V defaultValue)` | Returns the value to which the specified key is mapped, or defaultValue if this map contains no mapping for the key. |           |
| `put(K key, V value)`                      | Associates the specified value with the specified key in this map. |           |
| `remove(Object key)`                       | Removes the mapping for the specified key from this map if present. |           |
| `remove(Object key, Object value)`         | Removes the entry for the specified key only if it is currently mapped to the specified value. |           |
| `replace(K key, V value)`                  | Replaces the entry for the specified key only if it is currently mapped to some value. |           |
| `replace(K key, V oldValue, V newValue)`   | Replaces the entry for the specified key only if currently mapped to the specified value. |           |
| `entrySet()`                               | Returns a Set view of the mappings contained in this map.    |           |
| `keySet()`                                 | Returns a Set view of the keys contained in this map.        |           |
| `values()`                                 | Returns a Collection view of the values contained in this map. |           |
| `size()`                                   | Returns the number of key-value mappings in this map.        |           |
| `isEmpty()`                                | Returns true if this map contains no key-value mappings.     |           |

### TreeMap

`public interface Map<K,V>`

It is implemented using a Red-Black tree.TreeMap provides an ordered collection of key-value pairs, where the keys are ordered based on their natural order or a custom Comparator passed to the constructor.







## Tree



## Graph



## Comparable

`public interface Comparable<T>`

his interface imposes a total ordering on the objects of each class that implements it. This ordering is referred to as the class's natural ordering, and the class's compareTo method is referred to as its natural comparison method.

| Method               | Description                                                  | Exception |
| -------------------- | ------------------------------------------------------------ | --------- |
| `int compareTo(T o)` | Compares this object with the specified object for order. Returns a negative integer, zero, or a positive integer as this object is less than, equal to, or greater than the specified object |           |



## Comparator

```
@FunctionalInterface
public interface Comparator<T>
```

A comparison function, which imposes a total ordering on some collection of objects. Comparators can be passed to a sort method (such as `Collections.sort` or `Arrays.sort`) to allow precise control over the sort order. Comparators can also be used to control the order of certain data structures (such as sorted sets or sorted maps), or to provide an ordering for collections of objects that don't have a natural ordering.

This is a functional interface and can therefore be used as the assignment target for a lambda expression or method reference.

| Method                | Description                           | Exception |
| --------------------- | ------------------------------------- | --------- |
| `compare(T o1, T o2)` | Compares its two arguments for order. |           |



## Object

| Method                       | Description                                                 | Exception |
| ---------------------------- | ----------------------------------------------------------- | --------- |
| `boolean equals(Object obj)` | Indicates whether some other object is "equal to" this one. |           |
| `int hashCode()`             | Returns a hash code value for the object.                   |           |
| `String toString()`          | Returns a string representation of the object.              |           |



## Sort

Collections.Sort

Collections.reverseOrder()



## Generic

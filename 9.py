def simulate_fa(input_string):
    """
    Simulates a finite automaton that recognizes the specific string '101011'
    """
    # Define states
    # q0: Initial state
    # q1: Seen '1'
    # q2: Seen '10'
    # q3: Seen '101'
    # q4: Seen '1010'
    # q5: Seen '10101'
    # q6: Seen '101011' (accepting state)
    # q7: Error state (mismatch)
    
    current_state = 'q0'
    
    print(f"Input: {input_string}")
    print("Simulation:")
    
    transitions = {
        'q0': {'1': 'q1', '0': 'q7'},
        'q1': {'0': 'q2', '1': 'q7'},
        'q2': {'1': 'q3', '0': 'q7'},
        'q3': {'0': 'q4', '1': 'q7'},
        'q4': {'1': 'q5', '0': 'q7'},
        'q5': {'1': 'q6', '0': 'q7'},
        'q6': {'0': 'q7', '1': 'q7'},  # No valid transitions from accepting state
        'q7': {'0': 'q7', '1': 'q7'}   # Error state
    }
    
    # Process each character
    for i, char in enumerate(input_string):
        print(f"  State: {current_state}, Reading: {char}", end=" -> ")
        
        if current_state in transitions and char in transitions[current_state]:
            current_state = transitions[current_state][char]
        else:
            current_state = 'q7'  # Error state
        
        print(f"New state: {current_state}")
    
    # String is accepted if we end in state q6
    accepts = current_state == 'q6'
    print(f"\nFinal state: {current_state}")
    print(f"String {'accepted' if accepts else 'rejected'}")
    return accepts

# Example usage
test_string = "101011"
simulate_fa(test_string)
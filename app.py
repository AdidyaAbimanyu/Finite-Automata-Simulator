from flask import Flask, render_template, request
from graphviz import Digraph
import re

app = Flask(__name__)

@app.route('/', methods=['GET']) 
def index():
    return render_template('index.html')

@app.route('/soal1', methods=['GET'])
def soal1():
    return render_template('index1.html')

@app.route('/soal1/submit1', methods=['GET', 'POST'])
def submit1():
    if request.method == 'POST':
        # Ambil data input dari html
        states = set(request.form['states'].split(","))
        alphabet = set(request.form['alphabet'].split(","))
        transitions = {}
        transitions_input = request.form.getlist('transitions')     
        for transition in transitions_input:
            source, symbol, destination = transition.split(",")
            transitions[(symbol.strip(), source.strip())] = transitions.get((symbol.strip(), source.strip()), set()) | {destination.strip()}
        start_state = request.form['start_state']
        accept_states = set(request.form['accept_states'].split(","))
        # Buat objek NFA dari data input
        nfa = NFA(states, alphabet, transitions, start_state, accept_states)
        # Konversi NFA menjadi DFA
        dfa = nfa.nfa_to_dfa()
        # Tampilkan tabel transisi DFA
        dfa_table = dfa.display_transition_table()
        return render_template('index1.html', dfa_table=dfa_table)
    return render_template('index1.html', dfa_table=None)

class NFA:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
    
    def epsilon_closure(self, states):
        closure = set()
        stack = list(states)
        while stack:
            state = stack.pop()
            closure.add(state)
            print(self.transitions)
            if ('ε', state) in self.transitions:
                for next_state in self.transitions[('ε', state)]:
                    if next_state not in closure:
                        stack.append(next_state)
                        closure.add(next_state)
        print(closure)
        return frozenset(closure)

    def move(self, states, symbol):
        reachable_states = set()
        for state in states:
            if (symbol, state) in self.transitions:
                reachable_states.update(self.transitions[(symbol, state)])
        return frozenset(reachable_states)

    def nfa_to_dfa(self):
        dfa_states = set()
        dfa_transitions = {}
        dfa_start_state = self.epsilon_closure({self.start_state})
        dfa_accept_states = set()
        unmarked_states = [dfa_start_state]

        while unmarked_states:
            current_state_set = unmarked_states.pop(0)
            dfa_states.add(current_state_set)

            for symbol in self.alphabet:
                next_state = self.move(current_state_set, symbol)
                epsilon_closure_next = self.epsilon_closure(next_state)

                if epsilon_closure_next:
                    dfa_transitions[(current_state_set, symbol)] = epsilon_closure_next

                    if epsilon_closure_next not in dfa_states:
                        unmarked_states.append(epsilon_closure_next)

            if any(state in self.accept_states for state in current_state_set):
                dfa_accept_states.add(current_state_set)

        return DFA(dfa_states, self.alphabet, dfa_transitions, dfa_start_state, dfa_accept_states)

class DFA:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states

    def display_transition_table(self):
        # Mengurutkan simbol alphabet sesuai input pengguna
        sorted_alphabet = sorted(self.alphabet)
        
        # Pisahkan initial state dan final state
        initial_state = self.start_state
        final_states = sorted(self.accept_states - {self.start_state})
        
        # Sort state untuk menempatkan initial state di awal dan final state di akhir
        sorted_states = [initial_state] + sorted(self.states - {initial_state} - self.accept_states) + final_states

        dfa_table = []
        dfa_table.append("Tabel Transisi DFA:")
        dfa_table.append("---------------------------------------------------------------")
        dfa_table.append("|   Keadaan   |  " + "  |  ".join(sorted_alphabet) + "  |")
        dfa_table.append("---------------------------------------------------------------")

        max_state_length = max(len(state) for state in sorted_states)
        for state in sorted_states:
            row = "|"
            if state == initial_state:
                row += " -> "
            elif state in final_states:
                row += "  * "
            else:
                row += "    "
    
            state_str = ', '.join(sorted(state))
            row += f" {state_str} ".ljust(max_state_length + 6) + "|"
            for symbol in sorted_alphabet:
                next_state = self.transitions.get((state, symbol), set())

                next_state_str = ', '.join(sorted(next_state))
                row += f" {next_state_str} ".ljust(len(symbol) + 6) + "|"
            dfa_table.append(row)
        dfa_table.append("---------------------------------------------------------------")
        return '\n'.join(dfa_table)

@app.route('/soal2')
def soal2():
    return render_template('index2.html')

@app.route('/soal2/submit2', methods=['POST'])
def submit2():
    class Type:
        SYMBOL = 1
        CONCAT = 2
        UNION  = 3
        KLEENE = 4

    class ExpressionTree:
        def __init__(self, _type, value=None):
            self._type = _type
            self.value = value
            self.left = None
            self.right = None

    def constructTree(regexp):
        stack = []
        for c in regexp:
            if c.isalpha():
                stack.append(ExpressionTree(Type.SYMBOL, c))
            else:
                if c == "+":
                    z = ExpressionTree(Type.UNION)
                    z.right = stack.pop()
                    z.left = stack.pop()
                elif c == ".":
                    z = ExpressionTree(Type.CONCAT)
                    z.right = stack.pop()
                    z.left = stack.pop()
                elif c == "*":
                    z = ExpressionTree(Type.KLEENE)
                    z.left = stack.pop()
                stack.append(z)
        return stack[0]

    def inorder(et):
        if et._type == Type.SYMBOL:
            return et.value
        elif et._type == Type.CONCAT:
            return inorder(et.left) + "." + inorder(et.right)
        elif et._type == Type.UNION:
            return inorder(et.left) + "+" + inorder(et.right)
        elif et._type == Type.KLEENE:
            return inorder(et.left) + "*"

    def higherPrecedence(a, b):
        p = ["+", ".", "*"]
        return p.index(a) > p.index(b)

    def postfix(regexp):
        temp = []
        for i in range(len(regexp)):
            if i != 0 and (regexp[i-1].isalpha() or regexp[i-1] == ")" or regexp[i-1] == "*") and (regexp[i].isalpha() or regexp[i] == "("):
                temp.append(".")
            temp.append(regexp[i])
        regexp = temp
        
        stack = []
        output = ""

        for c in regexp:
            if c.isalpha():
                output = output + c
                continue

            if c == ")":
                while len(stack) != 0 and stack[-1] != "(":
                    output = output + stack.pop()
                stack.pop()
            elif c == "(":
                stack.append(c)
            elif c == "*":
                output = output + c
            elif len(stack) == 0 or stack[-1] == "(" or higherPrecedence(c, stack[-1]):
                stack.append(c)
            else:
                while len(stack) != 0 and stack[-1] != "(" and not higherPrecedence(c, stack[-1]):
                    output = output + stack.pop()
                stack.append(c)

        while len(stack) != 0:
            output = output + stack.pop()

        return output

    class FiniteAutomataState:
        def __init__(self):
            self.next_state = {}

    def evalRegex(et):
        if et._type == Type.SYMBOL:
            return evalRegexSymbol(et)
        elif et._type == Type.CONCAT:
            return evalRegexConcat(et)
        elif et._type == Type.UNION:
            return evalRegexUnion(et)
        elif et._type == Type.KLEENE:
            return evalRegexKleene(et)

    def evalRegexSymbol(et):
        start_state = FiniteAutomataState()
        end_state   = FiniteAutomataState()
        
        start_state.next_state[et.value] = [end_state]
        return start_state, end_state

    def evalRegexConcat(et):
        left_nfa  = evalRegex(et.left)
        right_nfa = evalRegex(et.right)

        left_nfa[1].next_state['epsilon'] = [right_nfa[0]]
        return left_nfa[0], right_nfa[1]

    def evalRegexUnion(et):
        start_state = FiniteAutomataState()
        end_state   = FiniteAutomataState()

        up_nfa   = evalRegex(et.left)
        down_nfa = evalRegex(et.right)

        start_state.next_state['epsilon'] = [up_nfa[0], down_nfa[0]]
        up_nfa[1].next_state['epsilon'] = [end_state]
        down_nfa[1].next_state['epsilon'] = [end_state]

        return start_state, end_state

    def evalRegexKleene(et):
        start_state = FiniteAutomataState()
        end_state   = FiniteAutomataState()

        sub_nfa = evalRegex(et.left)

        start_state.next_state['epsilon'] = [sub_nfa[0], end_state]
        sub_nfa[1].next_state['epsilon'] = [sub_nfa[0], end_state]

        return start_state, end_state

    def printStateTransitions(state, states_done, symbol_table):
        if state in states_done:
            return

        states_done.append(state)

        for symbol in list(state.next_state):
            current_state_index = symbol_table[state]
            next_states = state.next_state[symbol]

            for next_state in next_states:
                if next_state not in symbol_table:
                    symbol_table[next_state] = len(symbol_table)
                next_state_index = "q" + str(symbol_table[next_state])

            for next_state in next_states:
                printStateTransitions(next_state, states_done, symbol_table)

    postfix_exp = request.form['regexp']
    pr = postfix(postfix_exp)
    et = constructTree(pr)
    fa = evalRegex(et)

    transitions = []
    symbol_table = {fa[0]: 0}
    printStateTransitions(fa[0], [], symbol_table)

    for state in symbol_table:
        current_state = "q" + str(symbol_table[state])
        for symbol in state.next_state:
            next_states = state.next_state[symbol]
            for next_state in next_states:
                next_state_index = "q" + str(symbol_table[next_state])
                transitions.append({'state': current_state, 'symbol': symbol, 'next_state': next_state_index})

    return render_template('index2.html', pr=pr, transitions=transitions)

@app.route('/soal3')
def soal3():
    return render_template('index3.html')

@app.route('/soal3/submit3', methods=['POST'])
def submit3():
    # Membuat kelas untuk menginisialisasi objek DFA dengan masing-masing parameter
    class DFA:
        def __init__(self, states, addSimbol, transitions, state_awal, state_final):
            self.states = states
            self.addSimbol = addSimbol
            self.transitions = transitions
            self.state_awal = state_awal
            self.state_final = state_final

    # Fungsi pada state untuk melakukan transisi dari state saat ini dengan simbol input yang diberikan
    def nextState(stateCurrent, simbol, transitions):
        return transitions.get(stateCurrent, {}).get(simbol)

    # Fungsi pada DFA untuk memeriksa apakah state tersebut ekuivalen atau tidak
    def stateEkuivalen(state1, state2, classEkuivalen):
        return classEkuivalen[state1][state2]

    # Fungsi untuk memeriksa DFA menerima string yang diinputkan user pada DFA sebelum minimalisasi dan setelah minimalisasi
    def inputanString(dfa, input_string):
        stateCurrent = dfa.state_awal
        for simbol in input_string:
            stateLanjutan = nextState(stateCurrent, simbol, dfa.transitions)
            if stateLanjutan is None:
                return False
            stateCurrent = stateLanjutan
        return stateCurrent in dfa.state_final

    # Fungsi untuk membuat grafik / representasi DFA menggunakan DOT language dan nanti hasilnya pada web merupakan SVG (hasil vektor)
    def cetakDFA(dfa):
        # Membuat objek Digraph dari grapviz
        dot = Digraph()
        dot.attr(rankdir='LR')

        # Menambahkan node untuk setiap state dalam DFA
        for state in dfa.states:
            if state in dfa.state_final:
                dot.node(state, shape='doublecircle') # State Final memiliki gambar dengan lingkaran ganda
            else:
                dot.node(state)

        # Menambahkan transisi antar state dalam DFA
        for start_state, transitions in dfa.transitions.items():
            for simbol, stateLanjutan in transitions.items():
                dot.edge(start_state, stateLanjutan, label=simbol)

        # Menambahkan tanda state awal dengan label start
        dot.attr('node', shape='none', label='start')
        dot.node('')
        dot.edge('', dfa.state_awal)

        return dot.pipe(format='svg').decode('utf-8')

    # Fungsi untuk menghapus state-state yang tidak berguna
    def hapusStateUnreach(objek, start, getIn=None):
        if getIn is None:
            getIn = set()
        getIn.add(start)
        
        for sideState in objek.transitions[start]:
            if objek.transitions[start][sideState] not in getIn:
                hapusStateUnreach(objek, objek.transitions[start][sideState], getIn)
    
        return list(sorted(getIn))

    # Fungsi untuk melakukan minimalisasi DFA
    def minimalisasiDFA(dfa):
        # Langkah 1 menghapus state-state yang tidak berguna dari DFA yang diinputkan user
        stateReach = hapusStateUnreach(dfa, dfa.state_awal)
        
        # Langkah 2 melakukan Inisialisasi kelas ekuivalen untuk setiap pasangan state
        classEkuivalen = {}
        for state1 in stateReach:
            classEkuivalen[state1] = {}
            for state2 in stateReach:
                classEkuivalen[state1][state2] = (state1 in dfa.state_final) == (state2 in dfa.state_final)

        # Langkah 3 memeriksa kelas ekuivalen untuk setiap pasangan state berdasarkan transisi yang diberikan user
        for state1 in stateReach:
            for state2 in stateReach:
                for simbol in sorted(dfa.addSimbol):
                    stateAfter1 = nextState(state1, simbol, dfa.transitions)
                    stateAfter2 = nextState(state2, simbol, dfa.transitions)
                    if (stateAfter1 is None and stateAfter2 is not None) or (stateAfter1 is not None and stateAfter2 is None):
                        classEkuivalen[state1][state2] = False
                        
        # Langkah 3 melakukan iterasi untuk memperbarui kelas ekuivalen hingga tidak ada perubahan
        while True:
            ubahState = False
            for state1 in stateReach:
                for state2 in stateReach:
                    if not classEkuivalen[state1][state2]:
                        continue
                    for simbol in sorted(dfa.addSimbol):
                        stateAfter1 = nextState(state1, simbol, dfa.transitions)
                        stateAfter2 = nextState(state2, simbol, dfa.transitions)
                        if not stateEkuivalen(stateAfter1, stateAfter2, classEkuivalen):
                            classEkuivalen[state1][state2] = False
                            ubahState = True
                            break
                    if ubahState:
                        break
                if ubahState:
                    break
            if not ubahState:
                break

        # Langkah 5 mengelompokkan state-state ekuivalen menjadi satu grup DFA yang sudah diminimalisasi
        groupEkuivalen = {}
        groupAddState = 0
        for state1 in stateReach:
            if state1 not in groupEkuivalen.keys():
                groupAddState += 1
                groupEkuivalen[state1] = state1
            for state2 in stateReach:
                if state1 != state2 and classEkuivalen[state1][state2]:
                    groupEkuivalen[state2] = groupEkuivalen[state1]

        # Langkah 6 memberi label/parameter baru untuk setiap state dalam grup ekuivalen yang baru
        stateBaru = {}
        for state in stateReach:
            stateBaru[state] = str(groupEkuivalen[state])

        # Langkah 7 memperbarui state, state final, dan transisi sesuai dengan DFA yang diminimalisasi
        hasilStateBaru = []
        stateFinalBaru = []
        transisiBaru = []
        for state in stateReach:
            hasilStateBaru.append(stateBaru[state])
            if state in dfa.state_final:
                stateFinalBaru.append(stateBaru[state])

        
        for state in stateReach:
            for simbol in sorted(dfa.addSimbol):
                stateLanjutan = dfa.transitions[state].get(simbol)
                if stateLanjutan:
                    stateLanjutanBaru = stateBaru[stateLanjutan]
                    transisi = (stateBaru[state], simbol, stateLanjutanBaru)
                    transisiBaru.append(transisi)

        # Langkah 8 mengubah format transisi menjadi sesuai dengan DFA yang diminimalisasi
        transisiModifikasi = {}
        for transisi in transisiBaru:
            stateStart, simbol, stateLanjutan = transisi
            if stateStart not in transisiModifikasi:
                transisiModifikasi[stateStart] = {}
            transisiModifikasi[stateStart][simbol] = stateLanjutan

        # Langkah 9 membuat DFA baru hasil/setelah minimalisasi
        dfaBaru = DFA(
            states=hasilStateBaru,
            addSimbol=sorted(dfa.addSimbol),
            transitions=transisiModifikasi,
            state_awal=str(stateBaru[dfa.state_awal]),
            state_final=stateFinalBaru
        )

        return dfaBaru
    
    # Kode untuk membuat DFA dari input form yang ada di HTML menggunakan data yang dimasukkan user
    # Input form user ini nantinya akan disimpan dalam masing-masing dfa[]
    dfa = {}
    dfa['states'] = request.form['states'].split()
    dfa['addSimbol'] = request.form['inputSimbol'].split()
    dfa['state_awal'] = request.form['stateAwal']
    dfa['state_final'] = request.form['stateFinal'].split()
    dfa['transitions'] = {}
    for state in dfa['states']:
        dfa['transitions'][state] = {}
        for simbol in dfa['addSimbol']:
            stateLanjutan = request.form.get(f'transitions_{state}_{simbol}')
            dfa['transitions'][state][simbol] = stateLanjutan

    # Kode untuk membuat objek DFA dari data yang dimasukkan user pada input form
    inputDFA = DFA(
        states=set(dfa['states']),
        addSimbol=set(dfa['addSimbol']),
        transitions=dfa['transitions'],
        state_awal=dfa['state_awal'],
        state_final=set(dfa['state_final'])
    )

    # Uji DFA sebelum minimalisasi
    input_string = request.form['inputString']
    hasilStringSebelum = "DFA menerima string yang diuji" if inputanString(inputDFA, input_string) else "DFA tidak menerima string yang diuji"

    # Minimalisasi DFA
    minimalDFA = minimalisasiDFA(inputDFA)

    # Uji DFA setelah minimalisasi
    hasilStringSesudah = "DFA menerima string yang diuji" if inputanString(minimalDFA, input_string) else "DFA tidak menerima string yang diuji"
    
    # Ambil nilai yang ingin ditampilkan
    hasilState = ", ".join(minimalDFA.states)
    hasilSimbol = ", ".join(minimalDFA.addSimbol)
    hasilTransisi = []
    for start_state, transitions in minimalDFA.transitions.items():
        for simbol, stateLanjutan in transitions.items():
            hasilTransisi.append([start_state, simbol, stateLanjutan])
    hasilStateAwal = minimalDFA.state_awal
    hasilStateFinal = ", ".join(minimalDFA.state_final)
    
    # Kode untuk membuat grafik / representasi DFA sebelum dan sesudah minimalisasi
    graphSebelumMinim = cetakDFA(inputDFA)
    graphSesudahMinim = cetakDFA(minimalDFA)

    # Kode untuk mengembalikan hasil DFA minimalisasi ke halaman HTML untuk ditampilkan
    return render_template('result3.html', hasilState=hasilState, hasilSimbol=hasilSimbol, 
        hasilTransisi=hasilTransisi, hasilStateAwal=hasilStateAwal, 
        hasilStateFinal=hasilStateFinal, hasilStringSebelum=hasilStringSebelum,
        hasilStringSesudah=hasilStringSesudah, graphSebelumMinim=graphSebelumMinim, graphSesudahMinim=graphSesudahMinim)


@app.route('/soal4')
def soal4():
    return render_template('index4.html')

@app.route('/soal4/submit4', methods=['POST'])
def submit4():
    dfa1 = {}
    dfa1['states'] = request.form['states1'].split()
    dfa1['symbols'] = request.form['symbol1'].split()
    dfa1['initial_state'] = request.form['initialState1']
    dfa1['final_states'] = request.form['finalStates1'].split()
    dfa1['transitions'] = {}
    for state in dfa1['states']:
        dfa1['transitions'][state] = {}
        for symbol in dfa1['symbols']:
            next_state = request.form.get(f'transitions1_{state}_{symbol}')
            dfa1['transitions'][state][symbol] = next_state
    
    dfa2 = {}
    dfa2['states'] = request.form['states2'].split()
    dfa2['symbols'] = request.form['symbol2'].split()
    dfa2['initial_state'] = request.form['initialState2']
    dfa2['final_states'] = request.form['finalStates2'].split()
    dfa2['transitions'] = {}
    for state in dfa2['states']:
        dfa2['transitions'][state] = {}
        for symbol in dfa2['symbols']:
            next_state = request.form.get(f'transitions2_{state}_{symbol}')
            dfa2['transitions'][state][symbol] = next_state

    graph1_dfano4 = VisualizeDFA(dfa1)
    graph2_dfano4 = VisualizeDFA(dfa2)

    result = equivalenceDFA(dfa1, dfa2)

    if result:
        result_message = "Kedua DFA Ekuivalen"
    else:
        result_message = "Kedua DFA Tidak Ekuivalen"

    return render_template('result4.html', result=result_message, graph1_dfano4=graph1_dfano4, graph2_dfano4=graph2_dfano4)

def VisualizeDFA(dfa):
    dot = Digraph()
    dot.attr(rankdir='LR')

    # Tambahkan node
    for state in dfa['states']:
        if state in dfa['final_states']:
            dot.node(state, shape='doublecircle')
        else:
            dot.node(state)

    # Tambahkan transisi
    for start_state, transitions in dfa['transitions'].items():
        for symbol, next_state in transitions.items():
            dot.edge(start_state, next_state, label=symbol)

    dot.attr('node', shape='none', label='start')
    dot.node('')
    dot.edge('', dfa['initial_state'])

    return dot.pipe(format='svg').decode('utf-8')

def get_next_state(current_state, symbol, transitions):
    return transitions.get(current_state, {}).get(symbol)

def are_states_equivalent(state1, state2, dfa1, dfa2, equivalent_table):
    return equivalent_table[state1][state2]

def equivalenceDFA(dfa1, dfa2):
    def initialize_equivalence_table(dfa1, dfa2):
        equivalent_table = {}
        for state1 in dfa1['states']:
            equivalent_table[state1] = {}
            for state2 in dfa2['states']:
                equivalent_table[state1][state2] = (state1 in dfa1['final_states']) == (state2 in dfa2['final_states'])
        return equivalent_table

    equivalent_table = initialize_equivalence_table(dfa1, dfa2)

    if not equivalent_table[dfa1['initial_state']][dfa2['initial_state']]:
        return False

    for state1 in dfa1['states']:
        for state2 in dfa2['states']:
            for symbol in dfa1['symbols']:
                next_state1 = get_next_state(state1, symbol, dfa1['transitions'])
                next_state2 = get_next_state(state2, symbol, dfa2['transitions'])
                if (next_state1 is None and next_state2 is not None) or (next_state1 is not None and next_state2 is None):
                    equivalent_table[state1][state2] = False

    while True:
        changed = False
        for state1 in dfa1['states']:
            for state2 in dfa2['states']:
                if not equivalent_table[state1][state2]:
                    continue
                for symbol in dfa1['symbols']:
                    next_state1 = get_next_state(state1, symbol, dfa1['transitions'])
                    next_state2 = get_next_state(state2, symbol, dfa2['transitions'])
                    if not are_states_equivalent(next_state1, next_state2, dfa1, dfa2, equivalent_table):
                        equivalent_table[state1][state2] = False
                        changed = True
                        break
            if changed:
                break
        if not changed:
            break

    for state1 in dfa1['states']:
        for state2 in dfa2['states']:
            if are_states_equivalent(state1, state2, dfa1, dfa2, equivalent_table):
                for symbol in dfa1['symbols']:
                    next_state1 = get_next_state(state1, symbol, dfa1['transitions'])
                    next_state2 = get_next_state(state2, symbol, dfa2['transitions'])
                    if not are_states_equivalent(next_state1, next_state2, dfa1, dfa2, equivalent_table):
                        return False
            else:
                if state1 in dfa1['final_states'] != state2 in dfa2['final_states']:
                    return False
    return True

@app.route('/soal5', methods=['GET'])
def soal5():
    return render_template('index5.html')

@app.route('/soal5/dfa')
def soal5dfa():
    return render_template('result5dfa.html')

@app.route('/soal5/dfa/submit', methods=['POST'])
def soal5dfasubmit():
    def convert(p, q):
        if q in t:
            li.append(k[s.index(p)][t.index(q)])
            return k[s.index(p)][t.index(q)]
        else:
            return None

    s = request.form['states'].split()
    t = request.form['symbol'].split()
    last = request.form['finalStates']
    
    k = [[0 for _ in range(len(t))] for _ in range(len(s))]
    for i in range(len(s)):
        for j in range(len(t)):
            k[i][j] = request.form.get(f'transitions_{s[i]}_{t[j]}')
    
    li = []
    start = s[0]
    q = request.form['string']
    for i in q:
        start = convert(start, i)
        if start is None:
            result = "String tidak berisi yang ditentukan di Finite Automata"
            break
    if start == last:
        result = "String diterima"
    else:
        result = "String tidak diterima"
            
    return render_template('result5dfa.html', result=result)

@app.route('/soal5/enfa')
def soal5enfa():
    return render_template('result5enfa.html')

@app.route('/soal5/enfa/submit', methods=['POST'])
def soal5enfasubmit():
    def epsilon_closure(states):
        closure = set()
        stack = list(states)
        while stack:
            current_state = stack.pop()
            closure.add(current_state)
            epsilon_transitions = k[s.index(current_state)][-1]
            stack.extend(epsilon_transitions - closure)
        return closure

    def convert(states, symbol):
        next_states = set()
        for state in states:
            transitions = k[s.index(state)][t.index(symbol)]
            next_states.update(transitions)
        epsilon_transitions = epsilon_closure(next_states)
        next_states.update(epsilon_transitions)
        return next_states

    s = request.form['states'].split()
    t = request.form['symbol'].split()
    last = request.form['finalStates']

    k = [[set() for _ in range(len(t) + 1)] for _ in range(len(s))]
    for i in range(len(s)):
        for j in range(len(t) + 1):
            if j == len(t):
                k[i][j] = set(request.form.get(f"transitions_{s[i]}_ε", "").split())
            else:
                k[i][j] = set(request.form.get(f"transitions_{s[i]}_{t[j]}", "").split())

    start = epsilon_closure({s[0]})
    regex_pattern = request.form['string']

    pattern = re.compile(regex_pattern)

    if pattern.fullmatch(''):
        result = "String kosong tidak diterima"
    elif pattern.fullmatch('ε'):
        start = epsilon_closure(start)
    elif not pattern.fullmatch('ε'):
        for symbol in pattern.pattern:
            start = convert(start, symbol)
            if not start:
                result = "String tidak berisi yang ditentukan di Finite Automata"
                break

    if any(state in last for state in start):
        result = "String diterima"
    else:
        result = "String tidak diterima"
    return render_template('result5enfa.html', result=result)

@app.route('/soal5/nfa')
def soal5nfa():
    return render_template('result5nfa.html')

@app.route('/soal5/nfa/submit', methods=['POST'])
def soal5nfasubmit():
    def convert(p, q):
        next_states = set()
        for state in p:
            transitions = k[state].get(q, set())
            next_states.update(transitions)
        return next_states

    s = request.form['states'].split()
    t = request.form['symbol'].split()
    last = set(request.form['finalStates'].split())
    
    k = {}
    for state in s:
        k[state] = {}
        for symbol in t:
            transitions = set(request.form.get(f'transitions_{state}_{symbol}', '').split())
            k[state][symbol] = transitions

    start = {s[0]} 
    q = request.form['string']
    for i in q:
        start = convert(start, i)
        if not start:
            result = "String tidak berisi yang ditentukan di Finite Automata"
            break
    else:
        if any(state in last for state in start):
            result = "String diterima"
        else:
            result = "String tidak diterima"
        
    return render_template('result5nfa.html', result=result)

@app.route('/soal5/regex')
def soal5regex():
    return render_template('result5regex.html')

@app.route('/soal5/regex/submit', methods=['POST'])
def regexsubmit():
    regex_pattern = request.form['regex']
    string_to_check = request.form['string']
    if re.fullmatch(regex_pattern, string_to_check):
        result = "String diterima"
    else:
        result = "String tidak diterima"
    
    return render_template('result5regex.html', result = result)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Blueprint, render_template, request, current_app, jsonify, redirect, url_for, abort
import time, json, torch, os
from imt_environment.template import Template
from imt_environment.environment import State, logger

main = Blueprint("main", __name__)
translations = []
current_index = page_index = 0

success = []
editing_cost = []
turns = []
response_time = []
processes = []
process = []
last_hypothesis = None
state = State()
state.initialize_episode()
punc_transtab = str.maketrans(",;:!?()", "，；：！？（）")

@main.route("/", methods=["GET", "POST"])
def index():
    global current_index, page_index, state, success, editing_cost, turns, response_time, processes, translations, last_hypothesis
    testset = current_app.extensions["testset"]

    if request.method == "POST":
        submit_result = request.get_json()
        translation = submit_result['translation']
        template = submit_result["template"]
        if template:
            state.editing_cost += calc_editing_cost(template["tags"])
        process.append(template)
        process[-1]["hypothesis"] = last_hypothesis
        state.success = submit_result["success"]
        if state.success:
            logger.info(f"accept at turn {state.turn}!")
        else:
            logger.info("episode failed!")
        if (page_index < current_index):
            translations[page_index] = translation
            editing_cost[page_index] = state.editing_cost
            response_time[page_index]= state.response_time
            success[page_index]= state.success
            turns[page_index]= state.turn
            processes[page_index] = process
            redirect_url = url_for("main.submit", id=current_index)
        elif (current_index < len(testset)):
            translations.append(translation)
            editing_cost.append(state.editing_cost)
            response_time.append(state.response_time)
            success.append(state.success)
            turns.append(state.turn)
            processes.append(process)
            current_index += 1
            redirect_url = url_for("main.submit", id=current_index)
        else:
            redirect_url = url_for("main.complete")
        return jsonify({"url": redirect_url})
    elif request.method == "GET":
        checkpoint = current_app.extensions["args"].checkpoint
        if checkpoint is not None and os.path.exists(checkpoint) and not translations:
            state_dict = torch.load(checkpoint)
            current_index = state_dict["num"]
            success = state_dict["success"]
            turns = state_dict["turns"]
            editing_cost = state_dict["editing_cost"]
            response_time = state_dict["response_time"]
            translations = state_dict["translations"]
            processes = state_dict["processes"]

    return redirect(url_for("main.submit", id=current_index))

@main.route("/<int:id>", methods=["GET"])
def submit(id):
    global state, process, current_index, page_index, last_hypothesis
    if id > current_index:
        return redirect(url_for("main.submit", id=current_index))

    args = current_app.extensions["args"]
    testset = current_app.extensions["testset"]
    imt_system = current_app.extensions["imt_system"]

    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    src_path = args.src_path

    if (id < len(testset)):
        src_sentence = testset[id]
        page_index = id
    else:
        return redirect(url_for("main.complete"))

    state.initialize_episode()
    process = []
    state.turn = 1
    start_time = time.time()
    init_translation = imt_system.translate(src_sentence)
    respond_time = time.time() - start_time
    state.response_time += respond_time
    if tgt_lang == "zh":
        init_translation = init_translation.translate(punc_transtab)
    last_hypothesis = init_translation

    return render_template(
        "index.html",
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        src_path=src_path,
        sample_id=id,
        max_id=len(testset),
        src_sentence=src_sentence,
        init_translation=init_translation
    )

@main.route("/complete", methods=["GET"])
def complete():
    args = current_app.extensions["args"]
    testset = current_app.extensions["testset"]
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    src_path = args.src_path

    return render_template(
        "complete.html",
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        src_path=src_path,
        sample_id=len(testset),
        max_id=len(testset)
    )

@main.route("/translate", methods=["POST"])
def translate():
    global state, process, last_hypothesis
    req_data = request.get_json()
    imt_system = current_app.extensions["imt_system"]
    tgt_lang = current_app.extensions["args"].tgt_lang
    input_text = req_data["input_text"].strip()
    template, constraints, matched_pos = None, None, None
    cost = 0
    if req_data["template"] is not None:
        tags = req_data["template"]["tags"].copy()
        template = Template(req_data["template"]["revisedHypo"], tags)
        constraints = template.get_constraints()
        cost = calc_editing_cost(template.tag)

    if constraints:
        append_space = True
        i = 0
        while i < len(template.tag):
            if template.tag[i] == 0:
                i += 1
            elif template.tag[i] == 4:
                if tgt_lang != "zh":
                    append_space = True
                else:
                    append_space = False
                i += 1
            elif append_space:
                if template.revised_hypo[i].isalnum():
                    template.revised_hypo = template.revised_hypo[:i] + " " + template.revised_hypo[i:]
                    template.tag.insert(i, 1)
                    i += 1
                append_space = False
                i += 1
            else:
                i += 1

    start_time = time.time()
    translation = imt_system.translate(input_text, template)
    respond_time = time.time() - start_time

    state.response_time += respond_time
    state.turn += 1
    state.editing_cost += cost
    process.append(req_data["template"])
    process[-1]["hypothesis"] = last_hypothesis

    if tgt_lang == "zh":
        translation = translation.translate(punc_transtab)
    if constraints:
        cons_end_pos = 0
        matched_pos = []
        for constraint in constraints:
            constraint = constraint.strip()
            cons_begin_pos = translation.find(constraint, cons_end_pos)
            if cons_begin_pos >= 0:
                cons_end_pos = cons_begin_pos + len(constraint)
                matched_pos.append((cons_begin_pos, cons_end_pos))
    last_hypothesis = translation

    response = {"translated_text": translation, "constraints_pos": matched_pos}
    return jsonify(response)

@main.route('/export', methods=["GET"])
def export():
    output_stats = collapse()
    num = len(output_stats)
    success_rate = sum(success) / num
    avg_turns = sum(turns) / num
    avg_cost = sum(editing_cost) / num
    avg_response_time = sum(response_time) / num
    output_stats.append({
        "id": "avg statistics",
        "num": num,
        "avg_editing_cost": avg_cost,
        "avg_response_time": avg_response_time,
        "success_rate": success_rate,
        "avg_turns": avg_turns
    })
    logger.info("success rate: {:.3f} | avg turns: {:.2f} | avg editing cost: {:.2f} | avg responding time: {:.3f}".format(
        success_rate,
        avg_turns,
        avg_cost,
        avg_response_time
    ))
    path = current_app.extensions["args"].export_path
    if path is not None:
        try:
            with open(path, "w") as f:
                json.dump(output_stats, f)
            return "success"
        except Exception as e:
            print(e)
            abort(404)
    else:
        abort(404)

@main.route('/save', methods=["GET"])
def save():
    checkpoint = current_app.extensions["args"].checkpoint
    if checkpoint is not None:
        torch.save({
            "num": current_index,
            "translations": translations,
            "editing_cost": editing_cost,
            "success": success,
            "turns": turns,
            "response_time": response_time,
            "processes": processes
        }, checkpoint)
        return "success"
    else:
        abort(404)

def collapse():
    output_stats = []
    for i, (trans, cost, time, turn, suc, proc) in enumerate(zip(translations, editing_cost, response_time, turns, success, processes)):
        output_stats.append({
            "id": i,
            "translation": trans,
            "editing_cost": cost,
            "response_time": time,
            "success": suc,
            "turns": turn,
            "process": proc
        })
    return output_stats

def calc_editing_cost(tags):
    i, cost = 0, 0
    while (i < len(tags)):
        j = i + 1
        while j < len(tags) and tags[j] == tags[i]:
            j += 1
        if tags[i] == 0 or tags[i] == 4:
            cost += 1
        elif tags[i] == 2:
            cost += j - i
        elif tags[i] == 3:
            cost += j - i + 1
        i = j
    return cost

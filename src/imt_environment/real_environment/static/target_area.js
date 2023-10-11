var editable = undefined;

class OperationStack {
  constructor() {
    this.stack = [];
    this.index = -1;
  }

  add(state) {
    this.stack.splice(this.index + 1);
    this.stack.push(state);
    this.index = this.stack.length - 1;
  }

  get() {
    if (this.index > -1) {
      return this.stack[this.index];
    }
    return null;
  }

  clear() {
    this.stack = [];
    this.index = -1;
  }

  undo() {
    if (this.canUndo()) {
      this.index--;
      return this.stack[this.index];
    }
    return null;
  }

  redo() {
    if (this.canRedo()) {
      this.index++;
      return this.stack[this.index];
    }
    return null;
  }

  canUndo() {
    return this.index > 0;
  }

  canRedo() {
    return this.index < this.stack.length - 1;
  }
}

var operations = new OperationStack();
var compositionPos = null;

var isMac = false;
if (navigator.platform.startsWith("Mac")) {
  isMac = true;
}

function keydown(e) {
  if (e.key === "Backspace" && !compositionPos) {
    if ((isMac && e.metaKey && !e.shiftKey) || (!isMac && e.ctrlKey && !e.shiftKey)) {
      const selection = window.getSelection();
      const pos = getCursorPosition(selection.extentNode, selection.extentOffset);
      const lastTextNode = editable.lastChild.firstChild;
      if (selection.isCollapsed && pos < getCursorPosition(editable.lastChild, lastTextNode.length)) {
        const range = selection.getRangeAt(0);
        range.setEnd(lastTextNode, lastTextNode.length);
        const data = range.toString();
        if (selection.anchorNode.parentNode !== editable.lastChild) {
          editable.lastChild.remove();
        }
        range.deleteContents();
        if (pos == 0) {
          editable.firstChild.remove();
        } else {
          setCursorPosition([editable.textContent.length, editable.textContent.length]);
        }
        operations.add({ type: "deleteSpan", data: data, position: pos, length: data.length, preCursorPos: [pos, pos + data.length], postCursorPos: [pos, pos], content: editable.innerHTML });
      }
      e.preventDefault();
    } else if ((isMac && e.metaKey && e.shiftKey) || (!isMac && e.ctrlKey && e.shiftKey)) {
      const selection = window.getSelection();
      const pos = getCursorPosition(selection.extentNode, selection.extentOffset);
      const lastTextNode = editable.lastChild.firstChild;
      const isLastBlank = pos == editable.textContent.length - 1 && editable.lastChild.className == "blank";
      if (selection.isCollapsed && pos < editable.textContent.length && !isLastBlank) {
        const range = selection.getRangeAt(0);
        range.setEnd(lastTextNode, lastTextNode.length);
        const data = range.toString();
        const anchorParentNode = selection.anchorNode.parentNode;
        if (anchorParentNode.className == "blank" && selection.anchorOffset == 0) {
          range.setStart(selection.anchorNode, 1);
          pos = 1;
        }
        if (anchorParentNode !== editable.lastChild) {
          editable.lastChild.remove();
        }
        range.deleteContents();
        if (pos == 0) {
          editable.firstChild.remove();
        }
        if (anchorParentNode.className != "blank") {
          const blankSpan = document.createElement('span');
          blankSpan.className = "blank";
          blankSpan.appendChild(document.createTextNode('\u2003'));
          editable.appendChild(blankSpan);
          const prevSibling = blankSpan.previousSibling;
          if (prevSibling) {
            setRelativePosition(prevSibling.firstChild, prevSibling.firstChild.length, prevSibling.firstChild, prevSibling.firstChild.length);
          } else {
            setRelativePosition(blankSpan.firstChild, 0, blankSpan.firstChild, 0);
          }
          operations.add({ type: "replaceAsBlank", data: '\u2003', position: [pos, pos + data.length], length: 1, preCursorPos: [pos, pos + data.length], postCursorPos: [pos, pos], content: editable.innerHTML });
        } else {
          setRelativePosition(anchorParentNode.firstChild, 1, anchorParentNode.firstChild, 1);
          operations.add({ type: "replaceAsBlank", data: '', position: [pos, pos + data.length], length: 0, preCursorPos: [pos, pos + data.length], postCursorPos: [pos, pos], content: editable.innerHTML })
        }
      }
      e.preventDefault();
    } else {
      const selection = window.getSelection();
      if (selection.isCollapsed) {
        const pos = getCursorPosition(selection.extentNode, selection.extentOffset);
        if (pos == 0) {
          return;
        }
        const relativePos = selection.extentOffset;
        if (relativePos == 0) {
          prevSpan = selection.extentNode.parentNode.previousSibling;
          const data = prevSpan.textContent.slice(-1);
          operations.add({ type: "deleteChars", data: data, position: pos - 1, length: 1, preCursorPos: [pos, pos], postCursorPos: [pos - 1, pos - 1] });
        } else {
          const text = selection.extentNode.textContent;
          const data = text.charAt(relativePos - 1);
          if (!operations.canRedo() && operations.index > 0) {
            lastOperation = operations.get();
            if (lastOperation.type === "deleteChars" && lastOperation.position == pos) {
              lastOperation.data = data + lastOperation.data;
              lastOperation.position = pos - 1;
              lastOperation.length += 1;
              lastOperation.postCursorPos = [pos - 1, pos - 1];
            } else {
              operations.add({ type: "deleteChars", data: data, position: pos - 1, length: 1, preCursorPos: [pos, pos], postCursorPos: [pos - 1, pos - 1] });
            }
          } else {
            operations.add({ type: "deleteChars", data: data, position: pos - 1, length: 1, preCursorPos: [pos, pos], postCursorPos: [pos - 1, pos - 1] });
          }
        }
      } else {
        const range = selection.getRangeAt(0);
        const data = range.toString();
        const start = getCursorPosition(range.startContainer, range.startOffset);
        const end = start + data.length;
        range.setStart(range.startContainer, range.startOffset);
        range.setEnd(range.endContainer, range.endOffset);
        operations.add({ type: "deleteSpan", data: data, position: start, length: data.length, preCursorPos: [start, end], postCursorPos: [start, start] });
      }
    }
  } else if (e.key === "Delete") {
    e.preventDefault();
  } else if ((e.key == "z" || e.key == "Z") && ((isMac && e.metaKey) || (!isMac && e.ctrlKey)) && e.shiftKey) {
    const redoEvent = new InputEvent("beforeinput", {
      inputType: 'historyRedo',
      data: null,
      isComposing: false,
      cancelable: true
    });
    editable.dispatchEvent(redoEvent);
  } else if ((e.key == "z" || e.key == "Z") && ((isMac && e.metaKey) || (!isMac && e.ctrlKey))) {
    e.preventDefault();
    const undoEvent = new InputEvent("beforeinput", {
      inputType: 'historyUndo',
      data: null,
      isComposing: false,
      cancelable: true
    });
    editable.dispatchEvent(undoEvent);
  } else if (e.key == ' ' && e.shiftKey && !compositionPos) {
    const data = '\u2003';
    const selection = window.getSelection();
    if (selection.anchorNode === editable) {
      editable.innerHTML = `<span class=blank>${data}</span>`;
      setRelativePosition(editable.firstChild.firstChild, 1, editable.firstChild.firstChild, 1);
      operations.add({ type: "insertBlank", data: data, position: 0, length: 1, preCursorPos: [0, 0], postCursorPos: [1, 1], content: editable.innerHTML });
      e.preventDefault();
    } else if (selection.isCollapsed) {
      const relative_pos = selection.extentOffset;
      const parentNode = selection.extentNode.parentNode;
      const pos = getCursorPosition(selection.extentNode, relative_pos);
      if (parentNode.className != "blank" && (relative_pos < selection.extentNode.length || (!parentNode.nextSibling || parentNode.nextSibling.className !== "blank"))) {
        const insertedSpan = insertIntoSingleSpan(selection.extentNode, relative_pos, relative_pos, data, "blank");
        setRelativePosition(insertedSpan.firstChild, 1, insertedSpan.firstChild, 1);
        operations.add({ type: "insertBlank", data: data, position: pos, length: 1, preCursorPos: [pos, pos], postCursorPos: [pos + 1, pos + 1], content: editable.innerHTML });
      }
      e.preventDefault();
    } else {
      let range = selection.getRangeAt(0);
      const start_pos = getCursorPosition(range.startContainer, range.startOffset);
      const end_pos = getCursorPosition(range.endContainer, range.endOffset);
      if (selection.anchorNode.parentNode.className != "blank" && selection.anchorNode === selection.extentNode) {
        const insertedSpan = insertIntoSingleSpan(selection.anchorNode, range.startOffset, range.endOffset, data, "blank");
        setRelativePosition(insertedSpan.firstChild, 1, insertedSpan.firstChild, 1);
        let operation = { type: "replaceAsBlank", data: data, position: [start_pos, end_pos], length: 1, preCursorPos: [start_pos, end_pos], postCursorPos: [start_pos + 1, start_pos + 1] };
        if (insertedSpan.previousSibling && insertedSpan.previousSibling.className == "blank") {
          insertedSpan.previousSibling.remove();
          operation.data = '';
          operation.length = 0;
          operation.postCursorPos = [start_pos, start_pos];
        }
        if (insertedSpan.nextSibling && insertedSpan.nextSibling.className == "blank") {
          insertedSpan.nextSibling.remove();
          setRelativePosition(insertedSpan.firstChild, 0, insertedSpan.firstChild, 0);
          operation.data = '';
          operation.length = 0;
          operation.postCursorPos = [start_pos, start_pos];
        }
        operation.content = editable.innerHTML;
        operations.add(operation);
        e.preventDefault();
      } else if (selection.anchorNode.parentNode.className == "blank" && selection.anchorNode === selection.extentNode) {
        e.preventDefault();
      } else if (range.startContainer.parentNode.className == "blank") {
        const startContainer = range.startContainer.parentNode;
        const endContainer = range.endContainer.parentNode;
        if (range.endOffset == range.endContainer.length) {
          endContainer.remove();
        }
        range.deleteContents();
        startContainer.textContent = data;
        setRelativePosition(startContainer.firstChild, 1, startContainer.firstChild, 1);
        operations.add({ type: "replaceAsBlank", data: data, position: [start_pos, end_pos], length: 1, preCursorPos: [start_pos, end_pos], postCursorPos: [start_pos + 1, start_pos + 1], content: editable.innerHTML });
        e.preventDefault();
      } else if (range.endContainer.parentNode.className == "blank") {
        const endContainer = range.endContainer.parentNode;
        const startContainer = range.startContainer.parentNode;
        if (range.startOffset == 0) {
          startContainer.remove();
        }
        range.deleteContents();
        endContainer.textContent = data;
        setRelativePosition(endContainer.firstChild, 1, endContainer.firstChild, 1);
        operations.add({ type: "replaceAsBlank", data: data, position: [start_pos, end_pos], length: 1, preCursorPos: [start_pos, end_pos], postCursorPos: [start_pos + 1, start_pos + 1], content: editable.innerHTML });
        e.preventDefault();
      } else if (range.startContainer !== range.endContainer) {
        const insertedSpan = document.createElement('span');
        insertedSpan.className = "blank";
        insertedSpan.appendChild(document.createTextNode(data));
        const startContainer = range.startContainer.parentNode;
        const endContainer = range.endContainer.parentNode;
        const startOffset = range.startOffset;
        if (range.endOffset == range.endContainer.length) {
          endContainer.remove();
        }
        range.deleteContents();
        if (startOffset == 0) {
          editable.replaceChild(insertedSpan, startContainer);
        } else {
          editable.insertBefore(insertedSpan, startContainer.nextSibling);
        }
        setRelativePosition(insertedSpan.firstChild, 1, insertedSpan.firstChild, 1);
        let operation = { type: "replaceAsBlank", data: data, position: [start_pos, end_pos], length: 1, preCursorPos: [start_pos, end_pos], postCursorPos: [start_pos + 1, start_pos + 1] };
        if (insertedSpan.previousSibling && insertedSpan.previousSibling.className == "blank") {
          insertedSpan.previousSibling.remove();
          operation.data = '';
          operation.length = 0;
          operation.postCursorPos = [start_pos, start_pos];
        }
        if (insertedSpan.nextSibling && insertedSpan.nextSibling.className == "blank") {
          insertedSpan.nextSibling.remove();
          setRelativePosition(insertedSpan.firstChild, 0, insertedSpan.firstChild, 0);
          operation.data = '';
          operation.length = 0;
          operation.postCursorPos = [start_pos, start_pos];
        }
        operation.content = editable.innerHTML;
        operations.add(operation);
        e.preventDefault();
      }
    }
  }
}

function beforeInput(e) {
  if (e.inputType === "insertText") {
    const data = e.data;
    const dataLength = data.length;
    let selection = window.getSelection();
    if (selection.anchorNode === editable) {
      if (dataLength > 0) {
        editable.innerHTML = `<span class=insert>${data}</span>`;
        setRelativePosition(editable.firstChild.firstChild, dataLength, editable.firstChild.firstChild, dataLength);
        operations.add({ type: "insertChars", data: data, position: 0, length: dataLength, preCursorPos: [0, 0], postCursorPos: [dataLength, dataLength], content: editable.innerHTML })
        e.preventDefault();
      }
    } else if (selection.isCollapsed) {
      if (dataLength > 0) {
        const relative_pos = selection.extentOffset;
        const parentNode = selection.extentNode.parentNode;
        const pos = getCursorPosition(selection.extentNode, relative_pos);
        if (parentNode.nodeType != 3 && parentNode.className != "insert") {
          const insertedSpan = insertIntoSingleSpan(selection.extentNode, relative_pos, relative_pos, data, "insert");
          setRelativePosition(insertedSpan.firstChild, dataLength, insertedSpan.firstChild, dataLength);
          operations.add({ type: "insertChars", data: data, position: pos, length: dataLength, preCursorPos: [pos, pos], postCursorPos: [pos + dataLength, pos + dataLength], content: editable.innerHTML });
          e.preventDefault();
        } else {
          lastOperation = operations.get();
          if (lastOperation.type === "insertChars" && pos == lastOperation.position + lastOperation.length && !e.isComposing) {
            lastOperation.data += data;
            lastOperation.length += dataLength;
            lastOperation.postCursorPos = [pos + dataLength, pos + dataLength];
          } else if (e.isComposing) {
            selection.extentNode.textContent = selection.extentNode.textContent.slice(0, relative_pos) + data + selection.extentNode.textContent.slice(relative_pos);
            setRelativePosition(selection.extentNode, relative_pos + dataLength, selection.extentNode, relative_pos + dataLength);
            operations.add({ type: "insertChars", data: data, position: pos, length: dataLength, preCursorPos: [pos, pos], postCursorPos: [pos + dataLength, pos + dataLength], content: editable.innerHTML });
          } else {
            operations.add({ type: "insertChars", data: data, position: pos, length: dataLength, preCursorPos: [pos, pos], postCursorPos: [pos + dataLength, pos + dataLength] });
          }
        }
      }
    } else {
      let range = selection.getRangeAt(0);
      const start_pos = getCursorPosition(range.startContainer, range.startOffset);
      const end_pos = getCursorPosition(range.endContainer, range.endOffset);
      if (dataLength == 0) {
        const deletedData = range.toString()
        operations.add({ type: "deleteSpan", data: deletedData, position: start_pos, length: deletedData.length, preCursorPos: [start_pos, end_pos], postCursorPos: [start_pos, start_pos] });
        document.execCommand("delete", false);
      } else if (selection.anchorNode.parentNode.className != "insert" && selection.anchorNode === selection.extentNode) {
        const insertedSpan = insertIntoSingleSpan(selection.anchorNode, range.startOffset, range.endOffset, data, "insert");
        setRelativePosition(insertedSpan.firstChild, dataLength, insertedSpan.firstChild, dataLength);
        operations.add({ type: "replaceChars", data: data, position: [start_pos, end_pos], length: dataLength, preCursorPos: [start_pos, end_pos], postCursorPos: [start_pos + dataLength, start_pos + dataLength], content: editable.innerHTML });
        e.preventDefault();
      } else if (selection.anchorNode.parentNode.className == "insert" && selection.anchorNode === selection.extentNode) {
        if (!e.isComposing) {
          operations.add({ type: "replaceChars", data: data, position: [start_pos, end_pos], length: 1, preCursorPos: [start_pos, end_pos], postCursorPos: [start_pos + 1, start_pos + 1] });
        } else {
          const originText = selection.anchorNode.textContent;
          const startOffset = range.startOffset;
          selection.anchorNode.textContent = originText.slice(0, startOffset) + data + originText.slice(range.endOffset);
          setRelativePosition(selection.anchorNode, startOffset + dataLength, selection.anchorNode, startOffset + dataLength);
          operations.add({ type: "replaceChars", data: data, position: [start_pos, end_pos], length: dataLength, preCursorPos: [start_pos, end_pos], postCursorPos: [start_pos + dataLength, start_pos + dataLength], content: editable.innerHTML });
        }
      } else if (range.startContainer.parentNode.className == "insert") {
        if (range.startOffset == 0) {
          const startContainer = range.startContainer.parentNode;
          const endContainer = range.endContainer.parentNode;
          if (range.endOffset == range.endContainer.length) {
            endContainer.remove();
          }
          range.deleteContents();
          startContainer.textContent = data;
          setRelativePosition(startContainer.firstChild, dataLength, startContainer.firstChild, dataLength);
          operations.add({ type: "replaceChars", data: data, position: [start_pos, end_pos], length: dataLength, preCursorPos: [start_pos, end_pos], postCursorPos: [start_pos + dataLength, start_pos + dataLength], content: editable.innerHTML });
          e.preventDefault();
        } else {
          if (!e.isComposing) {
            operations.add({ type: "replaceChars", data: data, position: [start_pos, end_pos], length: 1, preCursorPos: [start_pos, end_pos], postCursorPos: [start_pos + 1, start_pos + 1] });
          } else {
            const startOffset = range.startOffset;
            const startContainer = range.startContainer.parentNode;
            const endContainer = range.endContainer.parentNode;
            if (range.endOffset == range.endContainer.length) {
              endContainer.remove();
            }
            range.deleteContents();
            startContainer.textContent += data;
            const newLength = startContainer.textContent.length;
            setRelativePosition(startContainer.firstChild, newLength, startContainer.firstChild, newLength);
            operations.add({ type: "replaceChars", data: data, position: [start_pos, end_pos], length: dataLength, preCursorPos: [start_pos, end_pos], postCursorPos: [start_pos + dataLength, start_pos + dataLength], content: editable.innerHTML });
          }
        }
      } else if (range.endContainer.parentNode.className == "insert") {
        const endContainer = range.endContainer.parentNode;
        const startContainer = range.startContainer.parentNode;
        if (range.startOffset == 0) {
          startContainer.remove();
        }
        range.deleteContents();
        endContainer.textContent = data + endContainer.textContent;
        setRelativePosition(endContainer.firstChild, dataLength, endContainer.firstChild, dataLength);
        operations.add({ type: "replaceChars", data: data, position: [start_pos, end_pos], length: dataLength, preCursorPos: [start_pos, end_pos], postCursorPos: [start_pos + dataLength, start_pos + dataLength], content: editable.innerHTML });
        e.preventDefault();
      } else if (range.startContainer !== range.endContainer) {
        const insertedSpan = document.createElement('span');
        insertedSpan.className = "insert";
        insertedSpan.appendChild(document.createTextNode(data));
        const startContainer = range.startContainer.parentNode;
        const endContainer = range.endContainer.parentNode;
        const startOffset = range.startOffset;
        if (range.endOffset == range.endContainer.length) {
          endContainer.remove();
        }
        range.deleteContents();
        if (startOffset == 0) {
          editable.replaceChild(insertedSpan, startContainer);
        } else {
          editable.insertBefore(insertedSpan, startContainer.nextSibling);
        }
        setRelativePosition(insertedSpan.firstChild, dataLength, insertedSpan.firstChild, dataLength);
        operations.add({ type: "replaceChars", data: data, position: [start_pos, end_pos], length: dataLength, preCursorPos: [start_pos, end_pos], postCursorPos: [start_pos + dataLength, start_pos + dataLength], content: editable.innerHTML });
        e.preventDefault();
      }
    }

  } else if (e.inputType == "historyUndo") {
    const currentState = operations.get();
    const lastState = operations.undo();
    if (lastState) {
      editable.innerHTML = lastState.content;
      if (lastState.content !== "<br>" && lastState.content !== "") {
        setCursorPosition(currentState.preCursorPos);
      }
    }
    e.preventDefault();
  } else if (e.inputType == "historyRedo") {
    const nextState = operations.redo();
    if (nextState) {
      editable.innerHTML = nextState.content;
      if (nextState.content !== "<br>" && nextState.content !== "") {
        setCursorPosition(nextState.postCursorPos);
      }
    }
    e.preventDefault();
  } else if (e.inputType === "insertCompositionText") {
    if (!compositionPos) {
      const range = window.getSelection().getRangeAt(0);
      const start = getCursorPosition(range.startContainer, range.startOffset);
      const end = getCursorPosition(range.endContainer, range.endOffset);
      compositionPos = [start, end];
    }
  } else if (e.inputType === "insertFromPaste") {
    e.preventDefault();
    const data = e.dataTransfer.getData('text/plain')
    if (data.length > 0) {
      const inputEvent = new InputEvent("beforeinput", {
        inputType: 'insertText',
        data: data,
        isComposing: true,
        cancelable:true
      });
      editable.dispatchEvent(inputEvent);
    }
  } else if (e.inputType === "deleteSoftLineBackward" ||
    e.inputType === "deleteWordBackward" ||
    e.inputType === "deleteByCut" ||
    e.inputType === "insertParagraph" ||
    e.inputType === "insertLineBreak" ||
    e.inputType === "deleteByDrag" ||
    e.inputType === "insertFromDrop") {
    console.log(e);
    e.preventDefault();
  }
}

function inputContent(e) {
  if (e.inputType === "deleteContentBackward") {
    operations.get().content = editable.innerHTML;
    const selection = window.getSelection();
    const range = selection.getRangeAt(0);
    selection.removeAllRanges();
    selection.addRange(range);
  } else if (e.inputType == "insertText") {
    operations.get().content = editable.innerHTML;
  }
}

function getEntireString() {
  let t = "";
  for (let node of editable.childNodes) {
    t += node.textContent;
  }
  return t.trim();
}

function insertIntoSingleSpan(span, begin, end, data, className) {
  const insertedSpan = document.createElement('span');
  insertedSpan.className = className;
  insertedSpan.appendChild(document.createTextNode(data));
  const parentNode = span.parentNode;
  if (begin == 0 && end == span.length) {
    editable.replaceChild(insertedSpan, parentNode);
  } else if (begin == 0 && end == 0) {
    editable.insertBefore(insertedSpan, parentNode);
  } else if (begin == span.length && end == span.length) {
    editable.insertBefore(insertedSpan, parentNode.nextSibling);
  } else if (begin == 0) {
    span.textContent = span.textContent.slice(end);
    editable.insertBefore(insertedSpan, parentNode);
  } else if (end == span.length) {
    span.textContent = span.textContent.slice(0, begin);
    editable.insertBefore(insertedSpan, parentNode.nextSibling);
  } else {
    let beforeInsert = document.createElement('span');
    let afterInsert = document.createElement('span');
    beforeInsert.className = parentNode.className;
    afterInsert.className = parentNode.className;
    beforeInsert.appendChild(document.createTextNode(span.textContent.slice(0, begin)));
    afterInsert.appendChild(document.createTextNode(span.textContent.slice(end)));
    editable.replaceChild(afterInsert, parentNode);
    editable.insertBefore(beforeInsert, afterInsert);
    editable.insertBefore(insertedSpan, afterInsert);
  }
  return insertedSpan;
}

function getCursorPosition(selectedNode, offset) {
  let pos = 0;
  for (let node of editable.childNodes) {
    const currentNode = node.nodeType == 1 ? node.firstChild : node;
    if (currentNode === selectedNode) {
      pos += offset;
      break;
    } else {
      pos += node.textContent.length;
    }
  }
  return pos;
}

function setCursorPosition(offset) {
  let pos = 0;
  if (editable.textContent === "") {
    return;
  }
  var currentNode, begin, end, beginNode, endNode;
  var f = false;
  for (let node of editable.childNodes) {
    currentNode = node.nodeType == 1 ? node.firstChild : node;
    if (!f) {
      if (currentNode.textContent.length + pos < offset[0] || (currentNode.textContent.length + pos == offset[0] && offset[1] > offset[0])) {
        pos += currentNode.textContent.length;
      } else {
        begin = offset[0] - pos;
        beginNode = currentNode;
        f = true;
      }
    }
    if (f) {
      if (currentNode.textContent.length + pos < offset[1]) {
        pos += currentNode.textContent.length;
      } else {
        end = offset[1] - pos;
        endNode = currentNode;
        break;
      }
    }
  }
  setRelativePosition(beginNode, begin, endNode, end);
}

function setRelativePosition(beginNode, beginOffset, endNode, endOffset) {
  const selection = window.getSelection();
  const range = document.createRange();
  range.setStart(beginNode, beginOffset);
  if (beginNode === endNode && beginOffset === endOffset) {
    range.collapse(true);
  } else {
    range.setEnd(endNode, endOffset);
  }
  selection.removeAllRanges();
  selection.addRange(range);
}

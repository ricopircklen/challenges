function performGetRequest1() {
    var resultElement = document.getElementById('getResult1');
    resultElement.innerHTML = '';
    axios.get(`http://localhost:8080/api/todot`)
        .then(function (response) {
            resultElement.innerHTML = generateSuccessHTMLOutput(response);
        })
        .catch(function (error) {
            resultElement.innerHTML = generateErrorHTMLOutput(error);
        });
}

function generateSuccessHTMLOutput(response) {
    return ` ${response.data.map(function (todo) {
        return `
        <ul class="list-group">
        <li class="list-group-item d-flex justify-content-between align-items-center" style="width: 10cm; margin: auto">
        <span class="badge badge-pill badge-info">${todo.id}</span>
        ${todo.otsikko}
         </li>
        </ul>
        `
    }).join('')}`;
}

function generateErrorHTMLOutput(error) {
    return error
}

function performGetRequest2() {
    var resultElement = document.getElementById('getResult2');
    var todoId = document.getElementById('todoId').value;
    resultElement.innerHTML = '';

    axios.delete(`http://localhost:8080/api/todot/${todoId}`)
        .then(function (response) {
            console.log(response);
            resultElement.innerHTML = `Tehtävä ${todoId} on poistettu.`;
        })
        .catch(function (error) {
            console.log("Poisto ei onnistunut");
            resultElement.innerHTML = generateErrorHTMLOutput(error);
        });
}

document.getElementById('todoInputForm').addEventListener('submit', performPostRequest);

function performPostRequest(e) {
    var resultElement = document.getElementById('postResult');
    var todoTitle = document.getElementById('todoTitle').value;
    resultElement.innerHTML = '';

    axios.post('http://localhost:8080/api/todot', {
        otsikko: `${todoTitle}`,
    })
        .then(function (response) {
            resultElement.innerHTML = `Uusi ToDo '${todoTitle}' on lisätty.`;
        })
        .catch(function (error) {
            resultElement.innerHTML = generateErrorHTMLOutput(error);
        });

    e.preventDefault();
}

function clearOutput() {
    var resultElement = document.getElementById('getResult1');
    resultElement.innerHTML = '';
    var resultElement = document.getElementById('getResult2');
    resultElement.innerHTML = '';
    var resultElement = document.getElementById('postResult');
    resultElement.innerHTML = '';
}
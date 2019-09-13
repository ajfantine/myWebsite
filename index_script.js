
function plusDivs(n) {
  showDivs(slideIndex += n);
}

function currentDiv(n){
  showDivs(slideIndex = n);
}

function showDivs(n) {
  var i;
  var x = document.getElementsByClassName("home_slides");

  var dots = document.getElementsByClassName("dot");

  if (n > x.length) {slideIndex = 1}

  if (n < 1) {slideIndex = x.length}

  for (i = 0; i < x.length; i++) {
    x[i].style.display = "none";
  }
  for (i=0; i < dots.length; i++){
    dots[i].className = dots[i].className.replace(" w3-orange", "");
  }
  x[slideIndex-1].style.display = "block";
  dots[slideIndex-1].className += " w3-orange";
}

function menuFixed() {
  if (window.pageYOffset >= sticky) {
    menu.classList.add("menu-fixed")
  } else {
     menu.classList.remove("menu-fixed");
  }
}

function hide(){
  var menu = document.getElementById("menu-bar");
  menu.style.display = "none";
}

function print_loc(){
  var menu = document.getElementById("menu-bar");
  var anchor = document.getElementById("about-anchor");
  console.log(anchor.offsetTop);
}

$(document).ready(function() {
	// For nav bar items
	$(".nav-item").click(function(e) {
		var position;
		e.preventDefault();
		// jQuery Function Number 1
		var a_href = $(this).attr('href');
		if (a_href == "#about-section" || a_href == "#biography-section") {
			// jQuery Function Number 2
			position = $(a_href).offset().top;
		} else {
			position = $(a_href).offset().top - 60;
		}
    	$('html,body').animate({
        	scrollTop: position}, {speed: 800, easing: "linear"});
	});

	// For sticky nav bar
	var stickyNavTop = $('.navbar').offset().top;
	var resizetimer;
	var about = $('#about-section').offset().top;
	var biography = $('#biography-section').offset().top;
	var education = $('#education-section').offset().top;
	var industry = $('#industry-section').offset().top;
	var research = $('#research-section').offset().top;
	var projects = $('#projects-section').offset().top;
	var resources = $('#resources-section').offset().top;
	var current_navitem = $('a[href="#about-section"]');
	current_navitem.css("background-color", "#ddd");

	function stickyNav() {
		// jQuery Function Number 3
		var scrollTop = $(window).scrollTop();
		if (scrollTop >= stickyNavTop) { 
			// jQuery Function Number 4
			$('.navbar').addClass('sticky');
		} else {
			// jQuery Function Number 5
			$('.navbar').removeClass('sticky'); 
		}
	}
	// re-adjust stickyNavTop when window resizes
	$(window).on('resize', function(){
        clearTimeout(resizetimer)
        resizetimer = setTimeout(function() {
            $('.navbar').removeClass('sticky');
            stickyNavTop = $('.navbar').offset().top;
            about = $('#about-section').offset().top;
            biography = $('#biography-section').offset().top;
            education = $('#education-section').offset().top;
			industry = $('#industry-section').offset().top;
			research = $('#research-section').offset().top;
			projects = $('#projects-section').offset().top;
			resources = $('#resources-section').offset().top;
            stickyNav();
        }, 50)
     })

	// Highlight navbar item of current viewing section
	function highlight() {
		var scrollTop = $(window).scrollTop();
		var navitem;
		if (scrollTop >= about && scrollTop < biography) {
			navitem = $('a[href="#about-section"]');
		} else if (scrollTop >= biography && (scrollTop + 60) < education) {
			navitem = $('a[href="#biography-section"]');
		} else if ((scrollTop + 60) >= education && (scrollTop + 60) < industry) {
			navitem = $('a[href="#education-section"]');
		} else if ((scrollTop + 60) >= industry && (scrollTop + 60) < research) {
			navitem = $('a[href="#industry-section"]');
		} else if ((scrollTop + 60) >= research && (scrollTop + 60) < projects) {
			navitem = $('a[href="#research-section"]');
		} else if ((scrollTop + 60) >= projects && (scrollTop + 60) < resources) {
			navitem = $('a[href="#projects-section"]');
		} else {
			navitem = $('a[href="#resources-section"]');
		}
		if (current_navitem != navitem) {
			// jQuery Function Number 6
			current_navitem.css("background-color", "#EEEEEE");
			navitem.css("background-color", "#ddd");
			current_navitem = navitem;
		}
	}

	// Abstract window re-positioning
	$(".button-abstract").mouseover(function() {
		var scrollTop = $(window).scrollTop() + 60;
		var scrollBottom = $(window).scrollTop() + $(window).height() - 66;
		var abstractBubble = $(this).find(".abstract-bubble");
		var currentClass = abstractBubble.children().eq(0).attr("class");
		if ((scrollBottom - scrollTop) < abstractBubble.height()) {
			abstractBubble.children().eq(0).removeClass().addClass("middle-dark-tooltip");
			abstractBubble.children().eq(1).removeClass().addClass("middle-light-tooltip");
			abstractBubble.css("margin-top", "15px");
			abstractBubble.css("transform", "translate(0%, -50%)");
		} else if (abstractBubble.offset().top <= scrollTop) {
			abstractBubble.children().eq(0).removeClass().addClass("top-dark-tooltip");
			abstractBubble.children().eq(1).removeClass().addClass("top-light-tooltip");
			abstractBubble.css("margin-top", "0px");
			abstractBubble.css("transform", "translate(0%, -10px)");
		} else if (abstractBubble.offset().top + abstractBubble.height() >= scrollBottom) {
			abstractBubble.children().eq(0).removeClass().addClass("bottom-dark-tooltip");
			abstractBubble.children().eq(1).removeClass().addClass("bottom-light-tooltip");
			abstractBubble.css("margin-top", "30px");
			abstractBubble.css("transform", "translate(0%, -100%)");
		} 
		currentClass = abstractBubble.children().eq(0).attr("class");
		if ((currentClass == "top-dark-tooltip" && abstractBubble.offset().top - abstractBubble.height() / 2 > scrollTop) || 
			(currentClass == "bottom-dark-tooltip" && abstractBubble.offset().top + 3 * abstractBubble.height() / 2 < scrollBottom)) {
			abstractBubble.children().eq(0).removeClass().addClass("middle-dark-tooltip");
			abstractBubble.children().eq(1).removeClass().addClass("middle-light-tooltip");
			abstractBubble.css("margin-top", "15px");
			abstractBubble.css("transform", "translate(0%, -50%)");
		}
	});

	$(window).scroll(function() {
		stickyNav();
		highlight();
	});

	


})
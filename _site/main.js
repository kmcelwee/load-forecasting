$(document).ready(function() {
	$('.article_container').click(function() { 
		where_to="#" + this.id.substring(3);
		$('html, body').animate({scrollTop: $(where_to).offset().top - 20}, 500);
	})
	$('#top').click(function() {
		$('html, body').animate({scrollTop: 0}, 500);
	})
});